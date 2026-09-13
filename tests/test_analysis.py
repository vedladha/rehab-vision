from types import SimpleNamespace
import numpy as np
from src.analysis import analyze
from src.geometry import angle_degrees

CFG=SimpleNamespace(SMOOTH_SECONDS=.10,MIN_PHASE_SECONDS=.1,MIN_REP_SECONDS=.5,
                    AMPLITUDE_FRACTION=.18,RETURN_FRACTION=.35,
                    INTERPOLATION_MAX_SECONDS=.15,BALANCE_MIN_SECONDS=.35)

def track(n, hip_y=None):
    xy=np.zeros((n,17,2),float); xy[:,:,0]=.5; xy[:,:,1]=.4
    xy[:,11]=[.45,.5]; xy[:,12]=[.55,.5]; xy[:,13]=[.45,.7]; xy[:,14]=[.55,.7]; xy[:,15]=[.45,.9]; xy[:,16]=[.55,.9]
    xy[:,5]=[.45,.25]; xy[:,6]=[.55,.25]
    if hip_y is not None: xy[:,11,1]=xy[:,12,1]=hip_y
    valid=np.ones((n,17),bool)
    return {"xy":xy,"valid":valid,"observed":valid.copy(),"interpolated":np.zeros_like(valid),"confidence":np.full((n,17),np.nan)}

def cycles(reps,fps=20):
    one=np.r_[np.full(8,.42),np.linspace(.42,.68,8),np.full(4,.68),np.linspace(.68,.42,8),np.full(7,.42)]
    return np.tile(one,reps),fps

def test_squat_known_count_and_timing():
    y,fps=cycles(3); a=analyze(track(len(y),y),fps,"squat","side","right",CFG,640,480)
    assert sum(e.status=="complete" for e in a.events)==3
    assert all(e.metrics["descent_s"]>0 and e.metrics["ascent_s"]>0 for e in a.events)
    assert a.events[1].metrics["time_between_previous_s"]>=0

def test_partial_rep_at_clip_boundary_is_preserved():
    y,fps=cycles(1); y=np.r_[y,np.linspace(.42,.68,9)]
    a=analyze(track(len(y),y),fps,"squat","side","right",CFG,640,480)
    assert a.events[-1].status=="incomplete_clip_boundary"

def test_stationary_jitter_does_not_count():
    rng=np.random.default_rng(2); y=.5+rng.normal(0,.001,200)
    a=analyze(track(len(y),y),20,"squat","side","right",CFG,640,480)
    assert not any(e.status=="complete" for e in a.events)

def test_missing_joints_interrupt_rep():
    y,fps=cycles(1); t=track(len(y),y); t["valid"][12:21,[11,12,13,14,15,16]]=False; t["observed"]=t["valid"].copy()
    a=analyze(t,fps,"squat","side","right",CFG,640,480)
    assert any("tracking" in e.status for e in a.events)

def test_view_availability_and_leg_labels():
    y,fps=cycles(1); a=analyze(track(len(y),y),fps,"squat","front","right",CFG,640,480)
    assert not a.availability["projected_knee_flexion"] and a.availability["lateral_knee_movement"]
    t=track(80); t["xy"][:,16,0]=.55+.18*np.sin(np.linspace(0,2*np.pi,80))**2
    s=analyze(t,20,"skater_squat","front","right",CFG,640,480)
    assert all(e.leg=="right" for e in s.events)

def test_aspect_ratio_angle_calculation():
    a=[.25,.5]; b=[.5,.5]; c=[.5,.75]
    assert angle_degrees(a,b,c,800,400)==90
    assert angle_degrees(a,b,c,400,800)==90

def test_interrupted_step_sequence_preserved():
    t=track(90); t["xy"][:,15,0]=t["xy"][:,16,0]=.5
    for start in (10,30): t["xy"][start:start+5,[15,16],0]+=np.linspace(0,.2,5)[:,None]
    a=analyze(t,20,"side_step_balance","front","right",CFG,640,480)
    assert not any(e.status=="complete" for e in a.events)
