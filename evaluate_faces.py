"""Evaluate unchanged face models against labeled local images.

CSV columns: image,expected_id,group. Empty expected_id means unknown.
Each image must contain one person; paths are relative to the CSV directory.
"""
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
import cv2
from python_engine.arcface_recognizer import ArcFaceRecognizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--output', type=Path, default=Path('face_evaluation.json'))
    args = parser.parse_args()
    recognizer = ArcFaceRecognizer()
    totals, groups, details = Counter(), defaultdict(Counter), []
    with args.manifest.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        if not {'image','expected_id'} <= set(reader.fieldnames or []):
            parser.error('CSV requires image and expected_id columns.')
        for row in reader:
            image = cv2.imread(str(args.manifest.parent / row['image']))
            if image is None:
                raise ValueError(f"Cannot read {row['image']}")
            # One full frame and a full-size crop per image would be redundant.
            # This evaluates image-level face matching, not body Re-ID or tracking.
            matches = recognizer.detect_and_recognize(image)
            if len(matches) > 1:
                raise ValueError(f"Use single-person images: {row['image']}")
            prediction = matches[0]['student_id'] if matches else None
            expected = row['expected_id'].strip() or None
            outcome = ('correct_known' if prediction == expected else 'wrong_known') if prediction else ('missed_known' if expected else 'correct_unknown')
            totals[outcome] += 1
            groups[row.get('group') or 'all'][outcome] += 1
            details.append({'image':row['image'],'expected_id':expected,'predicted_id':prediction,
                            'outcome':outcome,'score':matches[0]['confidence'] if matches else None})
    report = {'counts':dict(totals),'groups':{key:dict(value) for key,value in groups.items()},'images':details,
              'scope':'Image-level matching with quality gates; does not measure temporal confirmation or Re-ID.'}
    args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({'counts':report['counts'],'groups':report['groups'],'report':str(args.output)},indent=2))


if __name__ == '__main__':
    main()
