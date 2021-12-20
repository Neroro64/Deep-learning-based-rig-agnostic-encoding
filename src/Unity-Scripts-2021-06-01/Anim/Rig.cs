using System;
using System.Collections.Generic;
using BioIK;
// using MiscUtil.Collections.Extensions;
using UnityEngine;
using Tools;
using Zenject;
using Factory;

namespace Anim
{
    [RequireComponent(typeof(BioIK.BioIK))]
    [RequireComponent(typeof(Recorder))]
    public class Rig : MonoBehaviour
    {
        public JointRecorder[] joints;
        public Target[] targets;
        public BioIK.BioIK ik;
        public Locomotion locomotion;
        
        [Inject]
        public MainSystem sys;

        private void Start()
        {
            ik = GetComponent<BioIK.BioIK>();
            // if (joints == null || targets == null || joints.Length == 0 ||targets.Length == 0)
            FindJoints();
        }

        private void FindJoints()
        {
            List<JointRecorder> j = new List<JointRecorder>();
            List<Target> t = new List<Target>();
            foreach (var bone in ik.Segments)
            {
                if (bone.Joint != null)
                {
                    var jj = bone.gameObject.GetComponent<JointRecorder>(); 
                    if (jj == null)
                        jj = bone.gameObject.AddComponent<JointRecorder>();
                    
                    j.Add(jj);
                    
                    if (jj.keyBone)
                        t.Add(jj.target.gameObject.GetComponent<Target>());
                }
            }

            joints = j.ToArray();
            targets = t.ToArray();
        }

        public void ResetPosture()
        {
            foreach (var jo in joints)
            {
                jo.Reset();
            }
        }

     
    }
}