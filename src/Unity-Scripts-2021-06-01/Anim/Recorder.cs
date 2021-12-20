using System;
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
        [Header("Prefabs")]
        public AnimationSequence prefabSequence;
        // public Frame prefabFrame;
        // public Data.Joint prefabJoint;
        
        public AnimationSequence clip;
        [FormerlySerializedAs("frames")] public int clipSize = 30;
        [SerializeField] bool replaying = false;
        [SerializeField] bool resetting = false;
        public bool twoFold = false;
        public bool usingTargetValue = false;

        private Rig _rig;
        public int remainingFrames = 0;
        public int contactingFrames = 0;
        
        public string datasetName;
        public int clipID;
        public float smoothing_factor = 0.1f;
        public float thresholdP = 0.1f;
        public float thresholdR = 10f;

        private void Start()
        {
            _rig = GetComponent<Rig>();
        }

        private System.Func<bool, bool> ifTrue = x => x;

        public void Record(int contactFrames = 5)
        {
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
                var targetRotations = new quaternion[_rig.targets.Length];
                for (int ii = 0; ii < _rig.targets.Length; ++ii)
                {
                    targetTransforms[ii] = _rig.targets[ii].transform.position;
                    targetRotations[ii] = _rig.targets[ii].transform.rotation;
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
                var targetRotations = new quaternion[_rig.targets.Length];
                for (int ii = 0; ii < _rig.targets.Length; ++ii)
                {
                    targetTransforms[ii] = _rig.targets[ii].transform.position;
                    targetRotations[ii] = _rig.targets[ii].transform.rotation;
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
                var targetRotations = new quaternion[_rig.targets.Length];
                for (int ii = 0; ii < _rig.targets.Length; ++ii)
                {
                    targetTransforms[ii] = _rig.targets[ii].transform.position;
                    targetRotations[ii] = _rig.targets[ii].transform.rotation;
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

        private Quaternion computeQuaternion(Vector3 forward, Vector3 forward2, Vector3 up, Vector3 up2)
        {
            
            // float forwardAng = smoothing_factor * Vector3.Angle(forward, forward2);
            // float upAng = smoothing_factor * Vector3.Angle(up, up2);

            // forward = Vector3.RotateTowards(forward, forward2, forwardAng, 0f);
            // up = Vector3.RotateTowards(up, up2, upAng, 0f);
            return Quaternion.LookRotation(forward, up);
        }
        private void FixedUpdate()
        {
            float3 newPos;
            Quaternion newRot;
            Vector3 forward, up;
            float forwardAng, upAng;
            Transform joT;
            JointRecorder jo;
            
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
                    int j = 0;
                    for (int i = 0; i < joints.Length; ++i)
                    {
                        if (i >= _rig.joints.Length)
                            break;
                        joT = _rig.joints[i].transform;
                        jo = _rig.joints[i];
                        newPos = transform.root.TransformPoint(joints[i].position);
                        float3 pos = joT.position;
                        if (Vector3.Distance(newPos, joT.position) > thresholdP)
                        {
                            // joT.position = smoothing_factor *  (pos + joints[i].velocity)
                                // + (1-smoothing_factor) * newPos;
                                // joT.position = smoothing_factor * newPos + (1 - smoothing_factor) * pos;
                                joT.position = newPos;
                        }
                                   
                        forward = transform.root.TransformDirection(joints[i].rotMat.c0); 
                        up = transform.root.TransformDirection(joints[i].rotMat.c1);

                        newRot = computeQuaternion(forward, joT.forward, up, joT.up);

                        if (Quaternion.Angle(newRot, joT.rotation) > thresholdR)
                        {
                            joT.rotation = newRot;
                        }

                        if (jo.keyBone)
                        {
                            jo.target.position =transform.root.TransformPoint(joints[i].cost.TargetPosition);
                            
                            forward = transform.root.TransformDirection(joints[i].cost.TargetRotation.c0); 
                            up = transform.root.TransformDirection(joints[i].cost.TargetRotation.c1);
                            if (forward != Vector3.zero && up != Vector3.zero)
                            {
                                newRot = Quaternion.LookRotation(forward, up);
                                jo.target.rotation = newRot;
                                
                            }
                        }
                    }
                    
                    
                }
                // for (int i = 0; i < _rig.targets.Length; ++i)
                // {
                    // _rig.targets[i].transform.position = clip.frames[clipSize-remainingFrames].targetsPositions[i];
                    // _rig.targets[i].transform.rotation = clip.frames[clipSize-remainingFrames].targetsRotations[i];
                // }


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
        public void Record(int[] posIDs, bool randomRotation=false)
        {
            int contactFrames = 5;
            int[] ptr = new int[_rig.joints.Length];
            int len = _rig.joints.Length;
            int[] remainingContactFrames = new int[len];

            _rig.ik.EnableObjective.Invoke(false);
            ManualReset();
            replaying = resetting = false;
            _rig.ik.EnableObjective.Invoke(true);

            JointRecorder jo;
            for (int j = 0; j < len; ++j)
            {
                jo = _rig.joints[j];
                if (jo.keyBone && jo.Target)
                    jo.Target.SetPosition(posIDs[ptr[j]++], randomRotation);
            }
            
            
            List<Frame> frames = new List<Frame>();
            List<float> timestamps = new List<float>();
            Data.Joint[] joints; 
            Data.Joint[] jointsPrev = null;
            
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
                        
                        if (joints[j].contact && jojo.Target != null)
                        {
                            if (remainingContactFrames[j] > 0)
                            {
                                --remainingContactFrames[j];
                                if (remainingContactFrames[j] == 0)
                                    remainingContactFrames[j] = -1;
                            }
                            else
                                remainingContactFrames[j] = contactFrames;
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
                        
                        if (joints[j].contact && jojo.Target != null)
                        {
                            if (remainingContactFrames[j] > 0)
                            {
                                --remainingContactFrames[j];
                                if (remainingContactFrames[j] == 0)
                                    remainingContactFrames[j] = -1;
                            }
                            else
                                remainingContactFrames[j] = contactFrames;
                        }
                    }
                }

                for (int j = 0; j < len; ++j)
                {
                    jo = _rig.joints[j];
                    if (remainingContactFrames[j] == -1)
                    {
                        if (jo.Target != null)
                        {
                            if (ptr[j] < posIDs.Length)
                                jo.Target.SetPosition(posIDs[ptr[j]++], randomRotation);
                            else
                                jo.Target.SetPosition(-1);
                        }
                        remainingContactFrames[j] = 0;
                    }
                }

                var targetTransforms = new float3[_rig.targets.Length];
                var targetRotations = new quaternion[_rig.targets.Length];
                for (int ii = 0; ii < _rig.targets.Length; ++ii)
                {
                    targetTransforms[ii] = _rig.targets[ii].transform.position;
                    targetRotations[ii] = _rig.targets[ii].transform.rotation;
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
    public void Record(int[] posIDs, bool randomRot, float amplitude, float frequency, float stepSize)
        {
            var locomotion = _rig.locomotion;
            locomotion.SetAmplitude(amplitude);
            locomotion.SetFrequency(frequency);
            locomotion.SetStepSize(stepSize);
            
            int contactFrames = 5;
            int[] ptr = new int[_rig.joints.Length];
            int len = _rig.joints.Length;
            int[] remainingContactFrames = new int[len];

            _rig.ik.EnableObjective.Invoke(false);
            ManualReset();
            replaying = resetting = false;
            _rig.ik.EnableObjective.Invoke(true);

            JointRecorder jo;
            for (int j = 0; j < len; ++j)
            {
                jo = _rig.joints[j];
                if (jo.keyBone && jo.Target)
                    jo.Target.SetPositionI(posIDs[ptr[j]++], randomRot);
            }
            
            
            List<Frame> frames = new List<Frame>();
            List<float> timestamps = new List<float>();
            Data.Joint[] joints; 
            Data.Joint[] jointsPrev = null;

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

                        if (joints[j].contact && jojo.Target != null)
                        {
                            if (remainingContactFrames[j] > 0)
                            {
                                --remainingContactFrames[j];
                                if (remainingContactFrames[j] == 0)
                                    remainingContactFrames[j] = -1;
                            }
                            else
                                remainingContactFrames[j] = contactFrames;
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

                        if (joints[j].contact && jojo.Target != null)
                        {
                            if (remainingContactFrames[j] > 0)
                            {
                                --remainingContactFrames[j];
                                if (remainingContactFrames[j] == 0)
                                    remainingContactFrames[j] = -1;
                            }
                            else
                                remainingContactFrames[j] = contactFrames;
                        }
                    }
                }

                for (int j = 0; j < len; ++j)
                {
                    jo = _rig.joints[j];
                    if (remainingContactFrames[j] == -1)
                    {
                        if (jo.Target != null)
                        {
                            if (ptr[j] < posIDs.Length)
                                jo.Target.SetPositionI(posIDs[ptr[j]++], randomRot);
                            else
                                jo.Target.SetPositionI(-1);
                        }

                        remainingContactFrames[j] = 0;
                    }
                }
            
                
                locomotion.Step();
                
                var targetTransforms = new float3[_rig.targets.Length];
                var targetRotations = new quaternion[_rig.targets.Length];
                for (int ii = 0; ii < _rig.targets.Length; ++ii)
                {
                    targetTransforms[ii] = _rig.targets[ii].transform.position;
                    targetRotations[ii] = _rig.targets[ii].transform.rotation;
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

        public void ManualReset()
        {
            for (int i = 0; i < _rig.joints.Length; ++i)
            {
                _rig.joints[i].joint.X.TargetValue = 0;
                _rig.joints[i].joint.Z.TargetValue = 0;
                _rig.joints[i].joint.Y.TargetValue = 0;
            }

            _rig.ik.ManualUpdate();

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
                    // _rig.ik.EnableObjective.Invoke(false);
                    contactingFrames = -1;
                }
            }
        }

        public void Load(SyntheticDataGenerator dataGenerator)
        {
            dataGenerator.LoadSequence(this, clipID.ToString(), datasetName);
        }

        public void LoadUsingFilePath(SyntheticDataGenerator dataGenerator)
        {
            dataGenerator.LoadSequence(this, datasetName+clipID.ToString()+".json");
        }
        public void ResetPose()
        {
            _rig.ResetPosture();
        }
    }
}