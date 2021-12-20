using UnityEngine;
using Unity.Mathematics;
using System.Collections;

namespace Anim.Numeric
{
    public class Fn 
    {
        public static Vector3 W2L(Vector3 p, Transform local, bool isDir=false)
        {
            if (isDir)
                return local.InverseTransformDirection(p);
            else
                return local.InverseTransformPoint(p);
        }
        public static Vector3 L2W(Vector3 p, Transform local, bool isDir=false)
        {
            if (isDir)
                return local.TransformDirection(p);
            else
                return local.TransformPoint(p);
        }

        // public static float3x3 rotMat2rot(float3x3 from, float3x3 to)
        // {
        //     return to - from;
        // }
        // public static float3x3 angularMom(Vector3 velocity, float3x3 rot){}
        // {
        //     
        // }
    }
}