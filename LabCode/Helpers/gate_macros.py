import numpy as np
from qm.qua import *


def phased_XZ(element, b, a, x, z,):

    b.frame_rotation_2pi(-a/2, element)
    if x != 0:
        # if x == 1:
        #     b.play('baked_op', element, amp=x/2)
        #     b.play('baked_op', element, amp=x/2)
        # else:
        b.play('baked_op', element, amp=x)
    b.frame_rotation_2pi(a/2, element)
    b.frame_rotation_2pi(z/2, element)

def UGate(element, b, theta, phi, lam, ): # definition comes from virtual Z paper

    b.frame_rotation_2pi(lam-.25, element)
    b.play('baked_op', element, amp=1/2)
    b.frame_rotation_2pi(0.5-theta, element)
    b.play('baked_op', element, amp=1/2)
    b.frame_rotation_2pi(phi-.25, element)


def PMW2(element, b, sigma, theta, omega):

    b.frame_rotation_2pi(-sigma, element)
    b.play('baked_op', element, amp=float(theta)),
    b.frame_rotation_2pi(sigma-0.5, element)
    b.play('baked_op', element, amp=float(omega))
    b.frame_rotation_2pi(0.5, element)

def CZ(element_1, element_2, coupler, z1, z2):

    play('rise', coupler['c_name'])
    play('iSWAP_const', coupler['c_name'])
    play('fall', coupler['c_name'])
    frame_rotation_2pi(z1, element_1['q_name'])
    frame_rotation_2pi(z2, element_2['q_name'])


def flat_top_CZ(coupler, clk):

    play('rise', coupler)
    play('iSWAP_const', coupler, duration=clk)
    play('fall', coupler)


def flat_top_iSWAP(coupler):

    play('rise', coupler['c_name'])
    play('iSWAP_const',coupler['c_name'])
    play('fall', coupler['c_name'])
    # align()

def flat_top_iSWAP_Z_corr(q1,q2,coupler,p1,p2):

    play('rise', coupler)
    play('iSWAP_const', coupler)
    play('fall', coupler)
    frame_rotation_2pi(p1, q1)
    frame_rotation_2pi(p2, q2)
    frame_rotation_2pi(p1-p2, coupler)

def flat_top_iSWAP_dg(coupler):
    play('rise'*amp(-1), coupler)
    play('iSWAP_const'*amp(-1), coupler)
    play('fall'*amp(-1), coupler)
    # align()
    # frame_rotation_2pi(p1, q1)
    # frame_rotation_2pi(p2, q2)
    # frame_rotation_2pi(p1 - p2, coupler)

def flat_top_iSWAP_dg_Z_corr(q1,q2,coupler,p1,p2):
    play('rise'*amp(-1), coupler)
    play('iSWAP_const'*amp(-1), coupler)
    play('fall'*amp(-1), coupler)
    frame_rotation_2pi(p1, q1)
    frame_rotation_2pi(p2, q2)
    frame_rotation_2pi(p1-p2, coupler)

def update_frame(element,frame,add):

    assign(frame,frame+add)
    frame_rotation_2pi(frame,element)


def clifford_1Q(s, element, b): #phased xz to cliffords

    if s == 0:
        # I
        pass
    if s == 1:
        # X180
        phased_XZ(element, b, 0, 1, 0,)
    if s == 2:
        # Y180
        phased_XZ(element, b, 0.5, 1, 0,)
    if s == 3:
        # Y180 X180
        phased_XZ(element, b, 0, 0, 1)
    if s == 4:
        # X90 Y90
        phased_XZ(element, b, 0.5, 0.5, -0.5)
    if s == 5:
        # X90 -Y90
        phased_XZ(element, b, 0.5, -0.5, 0.5)
    if s == 6:
        # -X90 Y90
        phased_XZ(element, b, 0.5, 0.5, 0.5)
    if s == 7:
        # -X90 -Y90
        phased_XZ(element, b, 0.5, -0.5, -0.5)
    if s == 8:
        # Y90 X90
        phased_XZ(element, b, 0, 0.5, 0.5)
    if s == 9:
        # Y90 -X90
        phased_XZ(element, b, 0, -0.5, -0.5)
    if s == 10:
        # -Y90 X90
        phased_XZ(element, b, 0, 0.5, -0.5)
    if s == 11:
        # -Y90 -X90
        phased_XZ(element, b, 0, -0.5, 0.5)
    if s == 12:
        # X90
        phased_XZ(element, b, 0, 0.5, 0)
    if s == 13:
        # -X90
        phased_XZ(element, b, 0, -0.5, 0)
    if s == 14:
        # Y90
        phased_XZ(element, b, 0.5, 0.5, 0)
    if s == 15:
        # -Y90
        phased_XZ(element, b, 0.5, -0.5, 0)
    if s == 16:
        # -X90 Y90 X90
        phased_XZ(element, b, 0, 0, 0.5)
    if s == 17:
        # -X90 -Y90 X90
        phased_XZ(element, b, 0, 0, -0.5)
    if s == 18:
        # X180 Y90
        phased_XZ(element, b, 0.5, 0.5, 1)
    if s == 19:
        # X180 -Y90
        phased_XZ(element, b, 0.5, -0.5, 1)
    if s == 20:
        # Y180 X90
        phased_XZ(element, b, 0, 0.5, 1)
    if s == 21:
        # Y180 -X90
        phased_XZ(element, b, 0, -0.5, 1)
    if s == 22:
        # X90 Y90 X90
        phased_XZ(element, b, 0.25, 1, 0)
    if s == 23:
        # -X90 Y90 -X90
        phased_XZ(element, b, -0.25, 1, 0)
