import argparse,csv,json
from pathlib import Path
def load(path):
    p=Path(path)
    if p.suffix==".csv": return list(csv.DictReader(p.open()))
    data=json.loads(p.read_text()); return data.get("events",data) if isinstance(data,dict) else data
def main():
    a=argparse.ArgumentParser(); a.add_argument("labels"); a.add_argument("predictions"); a.add_argument("--tolerance-s",type=float,default=.5); args=a.parse_args(); truth=load(args.labels); pred=load(args.predictions)
    used=set(); errors=[]
    for t in truth:
        ts=float(t["start_s"]); candidates=[(abs(float(p["start_s"])-ts),i,p) for i,p in enumerate(pred) if i not in used and p.get("status")=="complete"]
        if candidates:
            err,i,p=min(candidates)
            if err<=args.tolerance_s: used.add(i); errors.append(err)
    result={"labeled_count":len(truth),"predicted_complete_count":sum(p.get("status")=="complete" for p in pred),"count_error":sum(p.get("status")=="complete" for p in pred)-len(truth),"matched":len(errors),"missed":len(truth)-len(errors),"false_observations":sum(p.get("status")=="complete" for p in pred)-len(errors),"mean_start_timestamp_error_s":sum(errors)/len(errors) if errors else None}
    print(json.dumps(result,indent=2))
if __name__=="__main__": main()
