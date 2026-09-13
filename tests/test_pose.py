import numpy as np
from src.pose import unwrap,person_to_pose,select_track,interpolate_short_gaps

def person(track=1,missing=False):
    pts=[[.5,.5]]*17
    if missing: pts[13]=[0,0]
    return {"track_id":track,"bbox_xywh":[.2,.1,.5,.8],"kpts_xy":pts}

def test_mock_gateway_response_and_absent_confidence():
    payload={"data":[{"content":{"object":"vitpose_plus.pose.frames","items":[{"index":0,"persons":[person()]}]}}]}
    frames=unwrap(payload); poses,quality=select_track(frames,1)
    assert poses[0]["confidence"] is None and quality[0]["status"]=="observed"

def test_missing_joint_and_bounded_interpolation():
    poses=[person_to_pose(person()),None,person_to_pose(person()),None,None,None,person_to_pose(person())]
    t=interpolate_short_gaps(poses,10,.15)
    assert t["interpolated"][1].all()
    assert not t["valid"][4].any()

def test_additional_people_are_flagged():
    frames=[{"index":0,"persons":[person(1),person(2)]}]
    _,quality=select_track(frames,1)
    assert quality[0]["status"]=="multiple_people"

def test_left_right_values_are_not_swapped():
    p=person_to_pose(person()); p["xy"][15]=[.2,.8]; p["xy"][16]=[.8,.8]
    assert p["xy"][15,0] < p["xy"][16,0]
