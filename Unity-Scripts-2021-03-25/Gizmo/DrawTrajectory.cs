using UnityEngine;
using Drawing;
using Anim;
using Tools;
using Factory;
using Unity.Mathematics;

namespace Gizmo
{
    public class DrawTrajectory : MonoBehaviourGizmos
    {
        [SerializeField] private Recorder _recorder;
        [SerializeField] private Rig _rig;
        [SerializeField] private int window;
        [SerializeField] private bool keyJointOnly = true;

        [SerializeField] private float size = 0.05f;
        [SerializeField] private Color color = Color.white;
        public override void DrawGizmos()
        {
            if (GizmoContext.InSelection(this))
            {
                int currentFrame = _recorder.clipSize - _recorder.remainingFrames;
                if (_recorder.remainingFrames > 0)
                {
                    
                Anim.Data.Joint[] joints;
                Anim.Data.Joint[] joints2;
                // using (Draw.InLocalSpace(transform))
                // {
                    var lower = window;
                    if (currentFrame + window > _recorder.clipSize)
                        lower = _recorder.clipSize - currentFrame;
                    
                    for (int i = 1; i <= lower; ++i)
                    {
                        joints = _recorder.clip.frames[currentFrame + i-1].joints;
                        joints2 = _recorder.clip.frames[currentFrame + i].joints;
                        if (keyJointOnly)
                        {
                            for (int j = 0; j < joints.Length; ++j)
                            {
                                if (joints[j].key)
                                {
                                    // Draw.SolidBox(joints[j].position, float3.zero + size, color);
                                    // Draw.Arrow(joints[j].position, joints2[j].position, Vector3.up, size, color);
                                    Draw.WireSphere(joints[j].position, size, color);
                                    Draw.Line(joints[j].position, joints2[j].position, color);
                                }
                            }
                        }
                        else 
                        {
                            for (int j = 0; j < joints.Length; ++j)
                            {
                                // Draw.SolidBox(joints[j].position, float3.zero + size, color);
                                // Draw.Arrow(joints[j].position, joints2[j].position, Vector3.up, size, color);
                                Draw.WireSphere(joints[j].position, size, color);
                                Draw.Line(joints[j].position, joints2[j].position, color);
                            }
                        }
                    }
                }
            } 
        }
    }
}
