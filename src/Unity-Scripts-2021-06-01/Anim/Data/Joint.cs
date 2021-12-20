// using System;
// using UnityEngine;
// using Unity.Mathematics;
// using UnityEngine.Serialization;
//
// namespace Anim.Data
// {
//     [CreateAssetMenu(fileName = "Joint", menuName = "Joint", order = 0)]
//     [Serializable]
//     public class Joint : ScriptableObject
//     {
//         /*
//          * Contains joint info */
//         
//         // [RootSpace] Joint
//         public float3 position;
//         public float3 rotation;
//         public quaternion rotQ;
//         public float3 forward;
//         public float3 up;
//         public float3 right;
//
//         // [LocalSpace] From BioIK
//         [FormerlySerializedAs("X")] public BioIK.BioJoint.Motion x;
//         [FormerlySerializedAs("Y")] public BioIK.BioJoint.Motion y;
//         [FormerlySerializedAs("Z")] public BioIK.BioJoint.Motion z;
//         public float3 anchor;
//         public float3 orientation;
//         
//         // [GoalSpace] Goal info
//         public float3 tGoalPosition;
//         public float3 tGoalDirection;
//     
//         // [RootSpace] Goal
//         public Objective.Cost Cost;
//         
//         // [RootSpace] Contact info
//         public bool contact;
//     }
// }