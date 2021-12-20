using UnityEngine;
using Unity.Mathematics;
using System;

namespace Anim{
    [Serializable]
    public abstract class Objective
    {
        [Serializable]
        public abstract class Cost
        {
            public abstract float ReturnSingleValueCost();
        }

        public abstract Cost ComputeCost(Transform t);
    }

    [Serializable]
    public class DistanceObjective : Objective
    {
        [Serializable]
        public class DistanceCost : Objective.Cost
        {
            public float3 ToTarget;
            public float3 ToTargetRotation;
            public float ToTargetDistance;
            public float ToTargetRotationAngle;
                    
            public float3 TargetPosition;
            public float3x3 TargetRotation;
            

            public override float ReturnSingleValueCost()
            {
                return math.lengthsq(ToTarget);
            }
        }

        public override Cost ComputeCost(Transform t)
        {
            DistanceCost c = new DistanceCost()
            {
                ToTarget = t.position,
                TargetPosition = t.position,
                ToTargetRotation = t.rotation.eulerAngles,
                ToTargetDistance = t.position.magnitude,
                ToTargetRotationAngle = 1f,
                TargetRotation = new float3x3(t.right, t.up, t.forward)
            };
            return c;
        }

        public static DistanceCost ComputeCost(Transform from, Transform to, Transform root)
        {
            float3x3 fRot = new float3x3(root.InverseTransformDirection(from.right), 
                    root.InverseTransformDirection(from.up), 
                    root.InverseTransformDirection(from.forward));
            float3x3 tRot = new float3x3(root.InverseTransformDirection(to.right),
                    root.InverseTransformDirection(to.up), 
                    root.InverseTransformDirection(to.forward));
        
            DistanceCost c = new DistanceCost()
            {
                ToTarget = root.InverseTransformPoint(to.position - from.position),
                ToTargetRotation = new float3(
                    math.acos(math.dot(tRot.c0, fRot.c0)),
                    math.acos(math.dot(tRot.c1, fRot.c1)),
                    math.acos(math.dot(tRot.c2, fRot.c2))),
                ToTargetDistance = Vector3.Distance(to.position, from.position),
                ToTargetRotationAngle = 
                    math.acos(2 * math.pow(
                        math.dot(to.transform.rotation, from.transform.rotation),2) - 1f),
                TargetPosition = root.InverseTransformPoint(to.transform.position),
                TargetRotation = tRot
            };
            return c;
        }
    }
}
