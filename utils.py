import os
import random
import json


def get_image(img_path, seg_num):
    img_list = [os.path.join(img_path, img) for img in os.listdir(img_path)]
    sampled_images = random.sample(img_list, seg_num)
    return sampled_images


def create_question(question_id, image, Object, label, template):
    question = dict()
    question["question_id"] = question_id
    question["image"] = image
    template1 = template
    template2 = template.replace(" a {}", " an {}") if " a {}" in template else template
    if Object[0] not in ["a", "e", "i", "o", "u"]:
        question["text"] = template1.format(Object)
    elif Object[0] in ["a", "e", "i", "o", "u"]:
        question["text"] = template2.format(Object)
    question["label"] = label
    return question


def pope(ground_truth_objects, segment_results, sample_num, template, neg_strategy, save_path, dataset):
    question_list = []
    question_id = 1
    output_file = os.path.join(save_path, dataset + "_pope_" + neg_strategy + ".json")

    gt_objects_list = list(ground_truth_objects.keys())
    sorted_objects = sorted(ground_truth_objects.items(), key=lambda x: x[1], reverse=True)
    sorted_co_occur = compute_co_occurrence(segment_results, save_path, dataset)

    for image in segment_results:
        history_object_list = []

        # Positive sampling (strategy-specific, deterministic)
        rng = random.Random(f"{neg_strategy}:{image['image']}")
        obj_pool = list(image["objects"])
        rng.shuffle(obj_pool)
        pos_objects = []
        if len(image["objects"]) > 0:
            offset = {"random": 0, "popular": 1, "adversarial": 2}[neg_strategy]
            for i in range(sample_num):
                idx = (offset + i) % len(image["objects"])
                pos_objects.append(image["objects"][idx])

        for pos_object in pos_objects:
            history_object_list.append(pos_object)
            question = create_question(question_id, image['image'], pos_object, 'yes', template)
            question_list.append(question)
            question_id += 1

            # Negative sampling (random)
            if neg_strategy == "random":
                selected_object = random.choice(gt_objects_list)
                while selected_object in history_object_list or selected_object in image["objects"]:
                    selected_object = random.choice(gt_objects_list)
                history_object_list.append(selected_object)
                question = create_question(question_id, image["image"], selected_object, 'no', template)
                question_list.append(question)
                question_id += 1

            # Negative sampling (popular)
            elif neg_strategy == "popular":
                flag = 0
                for j in range(len(sorted_objects)):
                    selected_object = sorted_objects[j][0]
                    if selected_object not in history_object_list and selected_object not in image["objects"]:
                        history_object_list.append(selected_object)
                        question = create_question(question_id, image["image"], selected_object, 'no', template)
                        question_list.append(question)
                        question_id += 1
                        flag = 1
                        break

                # In case no object is selected
                if not flag:
                    while True:
                        selected_object = random.choice(gt_objects_list)
                        if selected_object not in history_object_list and selected_object not in image["objects"]:
                            history_object_list.append(selected_object)
                            question = create_question(question_id, image["image"], selected_object, 'no', template)
                            question_list.append(question)
                            question_id += 1
                            break

            # Negative sampling (Adversarial)
            elif neg_strategy == "adversarial":
                flag = 0
                for j in range(len(sorted_co_occur[pos_object])):
                    selected_object = sorted_co_occur[pos_object][j]
                    if selected_object not in history_object_list and selected_object not in image["objects"]:
                        history_object_list.append(selected_object)
                        question = create_question(question_id, image["image"], selected_object, 'no', template)
                        question_list.append(question)
                        question_id += 1
                        flag = 1
                        break

                if not flag:
                    while True:
                        selected_object = random.choice(gt_objects_list)
                        if selected_object not in history_object_list and selected_object not in image["objects"]:
                            history_object_list.append(selected_object)
                            question = create_question(question_id, image["image"], selected_object, 'no', template)
                            question_list.append(question)
                            question_id += 1
                            break

    with open(output_file, 'w') as f:
        for question in question_list:
            json_str = json.dumps(question)
            f.write(json_str + "\n")


def _pick_negative(neg_strategy, pos_object, forbidden, gt_objects_list,
                   sorted_objects, sorted_co_occur):
    """Pick one negative object for the given strategy, skipping `forbidden`
    (objects in the image OR already used by any subset for this image)."""
    if neg_strategy == "random":
        choices = [o for o in gt_objects_list if o not in forbidden]
        return random.choice(choices) if choices else None

    if neg_strategy == "popular":
        for o in sorted_objects:                 # most frequent first
            if o not in forbidden:
                return o
        choices = [o for o in gt_objects_list if o not in forbidden]
        return random.choice(choices) if choices else None

    if neg_strategy == "adversarial":
        for o in sorted_co_occur.get(pos_object, []):   # top co-occurring first
            if o not in forbidden:
                return o
        for o in sorted_objects:                 # fall back to popular-ish
            if o not in forbidden:
                return o
        choices = [o for o in gt_objects_list if o not in forbidden]
        return random.choice(choices) if choices else None

    raise ValueError(f"Unknown neg_strategy: {neg_strategy}")


def pope_disjoint(ground_truth_objects, segment_results, sample_num, template,
                  save_path, dataset, require_distinct_positives=True):
    """Generate random/popular/adversarial POPE together so that no (image,
    object) question is shared across the three subsets.

    A single `used` set per image is threaded through all three strategies, so
    every positive and negative object is unique to one subset for that image.
    """
    # Process most-constrained strategy first so it gets first pick of objects:
    # adversarial draws from a short co-occurrence list, popular from a frequency
    # ranking, random from the whole vocabulary. Files are still written for all
    # three regardless of this order.
    strategies = ["adversarial", "popular", "random"]

    gt_objects_list = list(ground_truth_objects.keys())
    sorted_objects = [o for o, _ in sorted(ground_truth_objects.items(),
                                           key=lambda x: x[1], reverse=True)]
    sorted_co_occur = compute_co_occurrence(segment_results, save_path, dataset)

    question_lists = {s: [] for s in strategies}
    question_id = {s: 1 for s in strategies}
    skipped = 0

    n_needed = len(strategies) * sample_num   # distinct positives required
    for image in segment_results:
        img_objs = list(image["objects"])

        # Give each subset its own positives. Distinct positives need at least
        # n_needed ground-truth objects; otherwise either skip or allow reuse.
        if require_distinct_positives and len(img_objs) < n_needed:
            skipped += 1
            continue

        rng = random.Random(f"positives:{image['image']}")
        shuffled = img_objs[:]
        rng.shuffle(shuffled)

        used = set()          # every object used for this image, any subset
        for s_idx, neg_strategy in enumerate(strategies):
            # Slice distinct positives per subset when we have enough objects,
            # else fall back to the first `sample_num` (may overlap).
            if len(shuffled) >= n_needed:
                start = s_idx * sample_num
                pos_objects = shuffled[start:start + sample_num]
            else:
                pos_objects = shuffled[:sample_num]

            for pos_object in pos_objects:
                used.add(pos_object)
                question_lists[neg_strategy].append(
                    create_question(question_id[neg_strategy], image["image"],
                                    pos_object, "yes", template))
                question_id[neg_strategy] += 1

                forbidden = set(img_objs) | used
                neg_object = _pick_negative(neg_strategy, pos_object, forbidden,
                                            gt_objects_list, sorted_objects,
                                            sorted_co_occur)
                if neg_object is None:
                    continue
                used.add(neg_object)
                question_lists[neg_strategy].append(
                    create_question(question_id[neg_strategy], image["image"],
                                    neg_object, "no", template))
                question_id[neg_strategy] += 1

    for neg_strategy in strategies:
        output_file = os.path.join(save_path,
                                   dataset + "_pope_" + neg_strategy + ".json")
        with open(output_file, "w") as f:
            for question in question_lists[neg_strategy]:
                f.write(json.dumps(question) + "\n")

    if skipped:
        print(f"pope_disjoint: skipped {skipped} images with < {n_needed} objects")
    return question_lists


def generate_ground_truth_objects(segment_results, save_path, dataset):
    gt_objects = dict()
    output_file = os.path.join(save_path, dataset + "_ground_truth_objects.json")

    for image in segment_results:
        seg = image['objects']
        for o in seg:
            if o not in gt_objects:
                gt_objects[o] = 1
            else:
                gt_objects[o] += 1

    with open(output_file, 'w') as f:
        json_str = json.dumps(gt_objects)
        f.write(json_str)

    return gt_objects


def compute_co_occurrence(segment_results, save_path, dataset):
    output_file = os.path.join(save_path, dataset + "_co_occur.json")
    co_occur = dict()

    for image in segment_results:
        objects = image["objects"]
        for o in objects:
            if o not in co_occur:
                co_occur[o] = dict()
            for other_o in objects:
                if o == other_o:
                    continue
                if other_o not in co_occur[o]:
                    co_occur[o][other_o] = 1
                else:
                    co_occur[o][other_o] += 1

    sorted_co_occur = dict()
    for o in co_occur:
        objects = co_occur[o]
        sorted_co_occur_objects = sorted(objects.items(), key=lambda x: x[1], reverse=True)
        sorted_co_occur[o] = [item[0] for item in sorted_co_occur_objects]

    with open(output_file, 'w') as f:
        json_str = json.dumps(sorted_co_occur)
        f.write(json_str)

    return sorted_co_occur
