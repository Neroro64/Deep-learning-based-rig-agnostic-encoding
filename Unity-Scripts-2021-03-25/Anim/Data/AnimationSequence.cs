using UnityEngine;
using Unity.Mathematics;
using System;

namespace Anim.Data
{
    [CreateAssetMenu(fileName = "sequence", menuName = "Sequence", order = 0)]
    [Serializable]
    public class AnimationSequence : ScriptableObject
    {
        public float[] timestamp;
        public Frame[] frames;
    }
    [Serializable]
    public class Frame  
    {
        /*
         * Contains frame info */
        public Joint[] joints;
        public float3[] targetsPositions;
        public float3[] targetsRotations;
    }
    [Serializable]
    public class Joint 
    {
        /*
         * Contains joint info */
        public bool key;
        public int isLeft;
        public int chainPos;

        public float geoDistance;
        public float geoDistanceNormalised;
       // [RootSpace] Joint
        public float3 position;
        public float3 rotEuler;
        public quaternion rotQuaternion;
        public float3x3 rotMat;

        public float totalMass;
        public float3x3 inertiaObj;
        public float3x3 inertia;

        public float3 velocity;
        public float3 angularVelocity;
        public float3 linearMomentum;
        public float3 angularMomentum;
        
        // [LocalSpace] From BioIK
        public BioIK.BioJoint.Motion x;
        public BioIK.BioJoint.Motion y;
        public BioIK.BioJoint.Motion z;
        
        // [GoalSpace] Goal info
        public float3 tGoalPosition;
        public float3 tGoalDirection;
    
        // [RootSpace] Goal
        public DistanceObjective.DistanceCost cost;
        
        // [RootSpace] Contact info
        public bool contact;
        public static bool Validate(Anim.Data.Joint jo){
            if (
                jo.x == null || jo.y == null || jo.z == null
                )
                return false;
            else
                return true;
        }
    }
}