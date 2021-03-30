using UnityEngine;
using Drawing;
using Anim;
using Tools;
using Factory;

namespace Gizmo
{
    public class DrawTarget : MonoBehaviourGizmos
    {
        [SerializeField] private Target _target;
        public override void DrawGizmos()
        {
            if (GizmoContext.InSelection(this))
            {
                    // Spawning zone
                    Draw.WireSphere(_target.spawnCenter, _target.spawnRadius, Color.green);
                    using (Draw.InLocalSpace(transform))
                    // Interaction radius
                        Draw.WireSphere(Vector3.zero, _target.interactionRadius, Color.red);
                
            } 
            
        }
    }
}