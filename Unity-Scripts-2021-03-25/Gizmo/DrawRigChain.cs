using System;
using UnityEngine;
using Drawing;
using Anim;
using Tools;
using Factory;

namespace Gizmo
{
    public class DrawRigChain : MonoBehaviourGizmos
    {
        [SerializeField] private Transform start, end;
        [SerializeField] private Vector3 rotation;
        public override void DrawGizmos()
        {
            if (GizmoContext.InSelection(this))
            {
                Draw.Line(start.position, end.position);
                Draw.Arrow((end.position-start.position)/2f, rotation);
                
            } 
            
        }
    }
}