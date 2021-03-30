using UnityEngine;
using Unity.Mathematics;
using System.Collections;

namespace Anim.Numeric
{
    public class Fn 
    {
        public static Vector3 W2L(Vector3 p, Transform local, bool isDir=false)
        {
            /*
             * Transfrom a vector in world space to local space
             */
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
    }
}