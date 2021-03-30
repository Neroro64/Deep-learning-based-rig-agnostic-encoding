using UnityEngine;
using Anim.Data;
using Zenject;
using System.Collections.Generic;
using BioIK;
using Factory;
using Unity.Mathematics;
using UnityEditor;
using UnityEngine.Serialization;
using System.Linq;
using Random = UnityEngine.Random;

namespace Anim
{
    public class Recorder : MonoBehaviour
    {
        /*
         * The component is attached to every rig.
         * It is used for recording and replaying the animations.
         */
        
        [Header("Prefabs")]
        public AnimationSequence prefabSequence;
        public AnimationSequence clip;  // Created animation is stored here.
        
        public int clipSize = 120;
        [SerializeField] private bool replaying = false;
        [SerializeField] private bool resetting = false;
        public bool twoFold = false;
        public bool usingTargetValue = false;

        private Rig _rig;
        public int remainingFrames = 0;
        public int contactingFrames = 0;
        
        public string datasetName;
        public int clipID;

        private void Start()
        {
            _rig = GetComponent<Rig>();
        }

        private System.Func<bool, bool> ifTrue = x => x;

        public void Record(int contactFrames = 5)
        {
            /*
             * Assumes the target is already placed.
             */
            replaying = resetting = false;
            _rig.ik.EnableObjective.Invoke(true);
            contactingFrames = 0;
            
            UnityEngine.Random.InitState(2021);

            List<Frame> frames = new List<Frame>();
            List<float> timestamps = new List<float>();
            Data.Joint[] joints;
            Data.Joint[] jointsPrev = null;
            int len = _rig.joints.Length;
            bool[] contacted = new bool[_rig.joints.Length];

            for (int i = 0; i < this.clipSize; ++i)
            {
                // Set joint data for each joint in the rig
                joints = new Data.Joint[len];
                if (i == 0)
                {
                    
                    for (int j = 0; j < len; ++j)
                    {
                        joints[j] = new Anim.Data.Joint(); 
                        var jojo = _rig.joints[j];
                        jojo.SetJointData(joints[j], null);
                       
                        if (joints[j].contact)
                        {
                            contacted[j] = true;
                        }
                    }
                }
                else
                {
                    for (int j = 0; j < len; ++j)
                    {
                        joints[j] = new Anim.Data.Joint();
                        var jojo = _rig.joints[j];
                        jojo.SetJointData(joints[j], jointsPrev[j]);

                        if (joints[j].contact)
                        {
                            contacted[j] = true;
                        }
                    }
                }

                if (contactingFrames > -1 && contacted.Count(ifTrue) >= _rig.targets.Length)
                    Contacted(contactFrames);

                if (twoFold && contactingFrames < 0)
                {
                    foreach (var jo in _rig.joints)
                    {
                        jo.joint.X.TargetValue = jo.joint.defaultX.TargetValue;
                        jo.joint.Y.TargetValue = jo.joint.defaultY.TargetValue;
                        jo.joint.Z.TargetValue = jo.joint.defaultZ.TargetValue;
                    }
                }
                
                var targetTransforms = new float3[_rig.targets.Length];
                var targetRotations = new float3[_rig.targets.Length];
                for (int ii = 0; ii < _rig.targets.Length; ++ii)
                {
                    targetTransforms[ii] = _rig.targets[ii].transform.position;
                    targetRotations[ii] = _rig.targets[ii].transform.rotation.eulerAngles;
                }
                // Set frame data
                var f = new Anim.Data.Frame(); 
                f.joints = joints;
                f.targetsPositions = targetTransforms;
                f.targetsRotations = targetRotations;
                jointsPrev = joints;
                // Append timestamp and frame data to the sequence
                timestamps.Add(i);
                frames.Add(f);
                
                // Use BioIK to compute the next pose
                _rig.ik.ManualUpdate();
            }

            if (clip == null)
                clip = Instantiate(prefabSequence);

            clip.frames = frames.ToArray();
            clip.timestamp = timestamps.ToArray();
            
            ResetPose();
        }
        public void Record(List<float3[]> path, float[] timings, int contactFrames)
        {
            /*
             * Introduces a random path for the key-joints for a random duration, before spawning the target.
            */
            var disableIK = true;
            contactingFrames = 0;
            _rig.ik.EnableObjective.Invoke(false);
            
            replaying = resetting = false;
            _rig.ik.EnableObjective.Invoke(true);
            
            List<Frame> frames = new List<Frame>();
            List<float> timestamps = new List<float>();
            Data.Joint[] joints; 
            Data.Joint[] jointsPrev = null;
            int len = _rig.joints.Length;
            
            
            int[] keyframes = new int[timings.Length];
            for (int i = 0; i < keyframes.Length; ++i){
                keyframes[i] = (int)(timings[i] * this.clipSize);
            }
            int keyframePtr = 0;
            
            for (int i = 0; i < this.clipSize; ++i)
            {
                // Set joint data for each joint in the rig
                joints = new Data.Joint[len];
                bool contacted = false;
                if (i == 0)
                {
                    
                    for (int j = 0; j < len; ++j)
                    {
                        joints[j] = new Anim.Data.Joint(); 
                        var jojo = _rig.joints[j];
                        jojo.SetJointData(joints[j], null);
                        
                        if (joints[j].contact)
                        {
                            contacted = true;
                        }
                        
                    }
                }
                else
                {
                    for (int j = 0; j < len; ++j)
                    {
                        joints[j] = new Anim.Data.Joint();
                        var jojo = _rig.joints[j];
                        jojo.SetJointData(joints[j], jointsPrev[j]);
                        
                        if (joints[j].contact)
                        {
                            contacted = true;
                        }
                    }
                }


                if (disableIK && i < keyframes[keyframePtr]){
                    int k = 0;
                    foreach (var jo in _rig.joints){
                        if (jo.keyBone){
                            var jojo = jo.joint;
                            jojo.X.TargetValue = path[keyframePtr][k].x;
                            jojo.Y.TargetValue = path[keyframePtr][k].y;
                            jojo.Z.TargetValue = path[keyframePtr][k].z;
                            ++k;
                        }
                    }
                }
                else
                {
                    if (++keyframePtr >= keyframes.Length && disableIK)
                    {
                        disableIK = false;
                        _rig.ik.EnableObjective.Invoke(true);
                    }
                    
                    if (contacted)
                        Contacted(contactFrames); 
                }
                
                if (twoFold && contactingFrames < 0)
                {
                    foreach (var jo in _rig.joints)
                    {
                        jo.joint.X.TargetValue = 0;
                        jo.joint.Y.TargetValue = 0;
                        jo.joint.Z.TargetValue = 0;
                    }
                }
                
                var targetTransforms = new float3[_rig.targets.Length];
                var targetRotations = new float3[_rig.targets.Length];
                for (int ii = 0; ii < _rig.targets.Length; ++ii)
                {
                    targetTransforms[ii] = _rig.targets[ii].transform.position;
                    targetRotations[ii] = _rig.targets[ii].transform.rotation.eulerAngles;
                }
                // Set frame data
                var f = new Anim.Data.Frame(); 
                f.joints = joints;
                f.targetsPositions = targetTransforms;
                f.targetsRotations = targetRotations;
                jointsPrev = joints;
                
                // Append timestamp and frame data to the sequence
                timestamps.Add(i);
                frames.Add(f);
                
                // Use BioIK to compute the next pose
                _rig.ik.ManualUpdate();
            }

            if (clip == null)
                clip = Instantiate(prefabSequence);

            clip.frames = frames.ToArray();
            clip.timestamp = timestamps.ToArray();

            // ResetPose();
        }

        public void Record(float amplitude, float frequency, float stepSize)
        {
            /*
             * For locomotion
             */
            _rig.ik.EnableObjective.Invoke(true);
            
            List<Frame> frames = new List<Frame>();
            List<float> timestamps = new List<float>();
            Data.Joint[] joints; 
            Data.Joint[] jointsPrev = null;
            int len = _rig.joints.Length;

            var locomotion = _rig.locomotion;
            locomotion.SetAmplitude(amplitude);
            locomotion.SetFrequency(frequency);
            locomotion.SetStepSize(stepSize);
            
            for (int i = 0; i < this.clipSize; ++i)
            {
                // Set joint data for each joint in the rig
                joints = new Data.Joint[len];
                bool contacted = false;
                if (i == 0)
                {
                    
                    for (int j = 0; j < len; ++j)
                    {
                        joints[j] = new Anim.Data.Joint(); 
                        var jojo = _rig.joints[j];
                        jojo.SetJointData(joints[j], null);
                    }
                }
                else
                {
                    for (int j = 0; j < len; ++j)
                    {
                        joints[j] = new Anim.Data.Joint();
                        var jojo = _rig.joints[j];
                        jojo.SetJointData(joints[j], jointsPrev[j]);
                    }
                }
                
                locomotion.Step();
                
                var targetTransforms = new float3[_rig.targets.Length];
                var targetRotations = new float3[_rig.targets.Length];
                for (int ii = 0; ii < _rig.targets.Length; ++ii)
                {
                    targetTransforms[ii] = _rig.targets[ii].transform.position;
                    targetRotations[ii] = _rig.targets[ii].transform.rotation.eulerAngles;
                }
                // Set frame data
                var f = new Anim.Data.Frame(); 
                f.joints = joints;
                f.targetsPositions = targetTransforms;
                f.targetsRotations = targetRotations;
                jointsPrev = joints;
                
                // Append timestamp and frame data to the sequence
                timestamps.Add(i);
                frames.Add(f);
                
                // Use BioIK to compute the next pose
                _rig.ik.ManualUpdate();
            }

            if (clip == null)
                clip = Instantiate(prefabSequence);

            clip.frames = frames.ToArray();
            clip.timestamp = timestamps.ToArray(); 
            
            _rig.locomotion.Reset();
            ResetPose();
            
        }
        private void FixedUpdate()
        {
            /*
             * This function runs at 60Hz.
             */
            if (replaying && remainingFrames > 0)
            {
                Data.Joint[] joints = clip.frames[clipSize-remainingFrames].joints;
                if (usingTargetValue)
                {
                    for (int i = 0; i < joints.Length; ++i)
                    {
                        _rig.joints[i].joint.X.TargetValue = joints[i].x.TargetValue;
                        _rig.joints[i].joint.Y.TargetValue = joints[i].y.TargetValue;
                        _rig.joints[i].joint.Z.TargetValue = joints[i].z.TargetValue;
                    }
                    _rig.ik.ManualUpdate();
                }
                else
                {
                    for (int i = 0; i < joints.Length; ++i)
                    {
                        _rig.joints[i].transform.position = _rig.transform.root.TransformPoint(joints[i].position);
                        _rig.joints[i].transform.rotation = Quaternion.Inverse(_rig.transform.root.rotation) * joints[i].rotQuaternion;
                    } 
                }
                for (int i = 0; i < _rig.targets.Length; ++i)
                {
                    _rig.targets[i].transform.position = clip.frames[clipSize-remainingFrames].targetsPositions[i];
                    _rig.targets[i].transform.rotation = Quaternion.Euler(clip.frames[clipSize-remainingFrames].targetsRotations[i]);
                }


                --remainingFrames;
                if (remainingFrames <= 0)
                    replaying = false;
            }
            if (resetting)
            {
                for (int i = 0; i < _rig.joints.Length; ++i)
                {
                    _rig.joints[i].joint.X.TargetValue = 0;
                    _rig.joints[i].joint.Z.TargetValue = 0;
                    _rig.joints[i].joint.Y.TargetValue = 0;
                }

                _rig.ik.ManualUpdate();
                
            }
        }

        public void Replay(bool pause=false, bool usingTarget = false)
        {
            usingTargetValue = usingTarget;
            replaying = true;
            resetting = false;
            remainingFrames = clipSize;
            _rig.ik.EnableObjective.Invoke(false);
            EditorApplication.isPaused = pause;

        }

        public void Reset()
        {
            resetting = true;
            replaying = false;
            _rig.ik.EnableObjective.Invoke(false);
        }

        private void Contacted(int contactFrames)
        {
            if (contactingFrames == 0)
                contactingFrames = contactFrames;
            else if (contactingFrames > 0)
            {
                --contactingFrames;
                if (contactingFrames == 0)
                {
                    _rig.ik.EnableObjective.Invoke(false);
                    contactingFrames = -1;
                }
            }
        }

        public void Load(SyntheticDataGenerator dataGenerator)
        {
            dataGenerator.LoadSequence(this, clipID.ToString(), datasetName);
        }

        public void ResetPose()
        {
            _rig.ResetPosture();
        }
    }
}