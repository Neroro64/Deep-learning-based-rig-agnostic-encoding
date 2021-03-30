using UnityEngine;
using Drawing;
using Anim;
using Tools;
using Factory;

namespace Gizmo
{
    public class DrawCost : MonoBehaviourGizmos
    {
        [SerializeField] private Target _target;
        [SerializeField] private float threshold;
        public override void DrawGizmos()
        {
            
            var diff = _target.transform.position - transform.position;
            Draw.Line(_target.transform.position, transform.position, Color.black);
            if (Vector3.Distance(_target.transform.position, transform.position) < threshold)
                Draw.Label2D(_target.transform.position + Vector3.up * 0.1f, $"{diff.x}, {diff.y}, {diff.z}", Color.green);
            else
                Draw.Label2D(_target.transform.position + Vector3.up * 0.1f, $"{diff.x}, {diff.y}, {diff.z}", Color.red);
             
            
        }
    }
}