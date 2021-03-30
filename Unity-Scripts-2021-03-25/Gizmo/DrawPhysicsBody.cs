using UnityEngine;
using Drawing;
using Anim;
using Tools;
using Factory;
using Unity.Mathematics;

namespace Gizmo
{
    public class DrawPhysicsBody : MonoBehaviourGizmos
    {
        public Vector3 center;
        public float radius;
        public float height;
        public float deg;
        public Color color;
        public override void DrawGizmos()
        {
            if (GizmoContext.InSelection(this)) 
                Draw.WireCylinder(transform.position + center,  Vector3.up, height, radius, color);
        }
        

    }
}