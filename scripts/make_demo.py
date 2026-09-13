"""Create a no-API synthetic squat and complete report for dashboard QA."""
from pathlib import Path
from types import SimpleNamespace
import cv2,numpy as np
import config as cfg
from src.analysis import analyze
from src.render import render
from src.report import write_reports,plot
from src.video import encode

def main():
    out=cfg.OUTPUT_DIR/"synthetic-demo"; out.mkdir(parents=True,exist_ok=True)
    fps=20; n=180; w,h=640,720; source=out/"synthetic_source.mp4"
    writer=cv2.VideoWriter(str(source),cv2.VideoWriter_fourcc(*"mp4v"),fps,(w,h))
    one=np.r_[np.full(10,.44),np.linspace(.44,.67,10),np.full(4,.67),np.linspace(.67,.44,10),np.full(11,.44)]; hips=np.tile(one,4)
    hips=np.pad(hips,(0,n-len(hips)),constant_values=.44)
    xy=np.zeros((n,17,2)); xy[:,:,0]=.5; xy[:,:,1]=.3
    for i,y in enumerate(hips):
        xy[i,5]=[.44,y-.25]; xy[i,6]=[.56,y-.25]; xy[i,11]=[.44,y]; xy[i,12]=[.56,y]
        bend=(y-.44)*.55; xy[i,13]=[.42-bend,y+.17]; xy[i,14]=[.58+bend,y+.17]; xy[i,15]=[.38,y+.37]; xy[i,16]=[.62,y+.37]
        frame=np.full((h,w,3),(58,62,68),np.uint8); cv2.rectangle(frame,(0,int(h*.86)),(w,h),(42,45,49),-1); cv2.putText(frame,"SYNTHETIC MOTION - NOT REAL FOOTAGE",(35,50),cv2.FONT_HERSHEY_SIMPLEX,.7,(230,230,230),2); writer.write(frame)
    writer.release(); valid=np.ones((n,17),bool); track={"xy":xy,"valid":valid,"observed":valid.copy(),"interpolated":np.zeros_like(valid),"confidence":np.full((n,17),np.nan)}
    result=analyze(track,fps,"squat","side","right",cfg,w,h); temp=out/"annotated.temp.mp4"; render(source,track,result,temp,cfg.PRESCRIPTION["squat"]); encode(temp,out/"annotated.mp4",20); temp.unlink()
    (out/"raw_pose_response.json").write_text('{"synthetic": true, "note": "No API call"}')
    write_reports(out,result,cfg.PRESCRIPTION["squat"],{"synthetic":True,"processing":{"api_round_trip_s":0},"reference_commit":"5e4b92195d16889c21cd5de55c9f14337d5d8542"}); plot(out,result); print(out)
if __name__=="__main__": main()
