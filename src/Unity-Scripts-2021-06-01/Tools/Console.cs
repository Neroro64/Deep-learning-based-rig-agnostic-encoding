using System;
using UnityEngine;
using QFSW.QC;
using System.Collections.Generic;
using BioIK;
using Anim;
using Unity.Mathematics;
using Drawing;
using Zenject;
using Factory;
using System.Collections;

namespace  Tools
{
    [CommandPrefix("test.")]
    public class Console : MonoBehaviour
    {
        [SerializeField] private BioIK.BioSegment[] segments;
        [SerializeField] private double3[] targetJoints;
        
        [Command]
        public static void Bioik(int option=0, bool printAll=true)
        {
            var body = FindObjectOfType<BioIK.BioIK>();
            switch (option)
            {
                case 0: // Test Iterating through the rig
                    Debug.Log($"# Segments: {body.Segments}");
                    foreach (var seg in body.Segments)
                    {
                        if (printAll)
                        {
                            Debug.Log(seg.Transform.gameObject.name);
                        }
                        else
                        {
                            if (seg.Joint != null)
                                Debug.Log(seg.Transform.gameObject.name);
                        }
                    }
                    break;
                case 1: // Test printing all the joint angles
                    foreach (var seg in body.Segments)
                    {
                        if (seg.Joint != null)
                        {
                            Debug.Log($"J: {seg.Transform.gameObject.name}");
                            Debug.Log($"{seg.Joint.X.CurrentValue}, {seg.Joint.Y.CurrentValue}, {seg.Joint.Z.CurrentValue}");
                        }
                    }
                    break;
                case 2: // Test storing the joint angles for manual config
                    List<BioIK.BioSegment> segs = new List<BioSegment>();
                    List<double3> vecs = new List<double3>();
                    
                    foreach (var seg in body.Segments)
                    {
                        if (seg.Joint != null)
                        {
                            segs.Add(seg);
                            vecs.Add(new double3(seg.Joint.X.CurrentValue,seg.Joint.Y.CurrentValue,seg.Joint.Z.CurrentValue));
                        }
                    }

                    Console con = FindObjectOfType<Console>();
                    con.segments = segs.ToArray();
                    con.targetJoints = vecs.ToArray();
                    break;
                case 3: // Set custom joint angles 
                    Console con1 = FindObjectOfType<Console>();
                    int i = 0;
                    foreach (var seg in body.Segments)
                    {
                        if (seg.Joint != null)
                        {
                            if (con1.targetJoints[i].x != 0)
                                seg.Joint.X.TargetValue = con1.targetJoints[i].x;
                            if (con1.targetJoints[i].y != 0)
                                seg.Joint.Y.TargetValue = con1.targetJoints[i].y;
                            if (con1.targetJoints[i].z != 0)
                                seg.Joint.Z.TargetValue = con1.targetJoints[i].z;
                            // seg.Joint.X.CurrentValue = con1.targetJoints[i].x;
                            // seg.Joint.Y.CurrentValue = con1.targetJoints[i].y;
                            // seg.Joint.Z.CurrentValue  = con1.targetJoints[i].z;
                            
                            ++i;
                        }
                    }                    
                    break;   
               case 4: // Print joint positions
                    foreach (var seg in body.Segments)
                    {
                        if (seg.Joint != null)
                        {
                           print($"{seg.gameObject.name}: {seg.Joint.GetAnchor()}");
                           print($"Pos: {seg.transform.position}");
                           print($"Root: {seg.transform.root.position}");
                           print($"PosRelativeToRoot: {seg.transform.root.InverseTransformPoint(seg.transform.position)}");
                        }
                    }                    
                    break; 
                case 5: // Print joint orientations
                    foreach (var seg in body.Segments)
                    {
                        if (seg.Joint != null)
                        {
                           print($"{seg.gameObject.name}: {seg.Joint.GetOrientation()}");
                           print($"Rot: {seg.transform.rotation.eulerAngles}");
                           print($"Root: {seg.transform.root.rotation.eulerAngles}");
                           print($"RotRelativeToRoot: {seg.transform.root.InverseTransformDirection(seg.transform.rotation.eulerAngles)}");
                        }
                    }                    
                    break;
                case 6: // Print joint orientations using two directions
                    foreach (var seg in body.Segments)
                    {
                        if (seg.Joint != null)
                        {
                           print($"{seg.gameObject.name}: {seg.Joint.GetOrientation()}");
                           print($"forward: {seg.transform.forward}");
                           print($"up: {seg.transform.up}");
                           print($"RotRelativeToRoot: {seg.transform.root.InverseTransformDirection(seg.transform.forward)}");
                        }
                    }                    
                    break;   
                case 7:
                    Rig[] rigs = FindObjectsOfType<Rig>();
                    foreach (var rig in rigs)
                    {
                        int leftsum = 0;
                        int chainsum = 0;
                        int levelsum = 0;
                        foreach (var joint in rig.joints)
                        {
                            leftsum += joint.isLeft;
                            chainsum += joint.chainPos;
                            levelsum += joint.level;
                        }
                        print($"leftSum: {leftsum}, chainSum: {chainsum}, levelSum: {levelsum}");
                    }

                    break;
            }
        }

        [Command]
        public static void DisplayInject()
        {
            var rig = FindObjectOfType<Anim.Rig>();
            Debug.Log(rig.sys.name);
        }

        [Command]
        public static void DrawCircle(float radius=2)
        {
            Draw.Circle(new float3(0,0,0), new float3(0,1,0), radius);
        }

        [Command]
        public static void Record()
        {
            var recorders = FindObjectsOfType<Anim.Recorder>();
            foreach(var recorder in recorders)
                recorder.Record();
        } 
        [Command]
        public static void RecordUsingPath(float t1 = 0.2f, float t2 = 0.5f)
        {
            var recorder = FindObjectOfType<Anim.Recorder>();
            var rig = recorder.gameObject.GetComponent<Rig>();
            var planner = FindObjectOfType<Planner>();

            var path = planner.GetRandomPath(rig);
            float[] timings = {t1, t2};
            recorder.Record(path, timings, planner.GetRandomContactingFrames());
        }  
        [Command]
        public static void Replay(bool pause=false, bool usingTarget=false)
        {
            var recorders = FindObjectsOfType<Anim.Recorder>();
            foreach(var recorder in recorders)
                recorder.Replay(pause, usingTarget);
        }
        [Command]
        public static void Reset()
        {
            var recorders = FindObjectsOfType<Anim.Recorder>();
            foreach(var recorder in recorders)
                recorder.Reset();
        }

        [Command]
        public static void RandomSpawnTarget(int i)
        {
            var targets = FindObjectsOfType<Target>();
            foreach (var t in targets)
            {
                if (i == 0)
                    t.RandomSpawn();
                else if (i==1)
                    t.RandomSpawn(Target.SpawnMode.OutsideSphere);
                else if (i==2)
                    t.RandomSpawn(Target.SpawnMode.OnPlane);
                else
                    t.RandomSpawn(Target.SpawnMode.OnSphere);
            }
        }

        [Command]
        public static void LoadSample(int i, string datasetName)
        {
            var _rig = GameObject.FindObjectOfType<Rig>();
            var recorder = _rig.GetComponent<Recorder>();
            var dataGenerator = GameObject.FindObjectOfType<SyntheticDataGenerator>();
            
            dataGenerator.LoadSequence(recorder, i.ToString(), datasetName);
            
        }

        [Command]
        public static void LoadSample(string filename)
        {
            var _rig = GameObject.FindObjectOfType<Rig>();
            var recorder = _rig.GetComponent<Recorder>();
            var dataGenerator = GameObject.FindObjectOfType<SyntheticDataGenerator>();
            
            dataGenerator.LoadSequence(recorder, filename, "");
            
        }

        [Command]
        public static void WalkStep()
        {
            var locomotion = GameObject.FindObjectOfType<Locomotion>();
            locomotion.Step();
        }

        [Command]
        public static void Walk(float t, float duration)
        {
            var locomotions = GameObject.FindObjectsOfType<Locomotion>();
            foreach(var loco in locomotions)
                loco.Walk(t, duration);
        }

        [Command]
        public static void ResetLocomotion()
        {
            var locomotions = GameObject.FindObjectsOfType<Locomotion>();
            foreach (var loco in locomotions)
                loco.Reset();
        }

        [Command]
        public static void DistributeAndActivate(bool activate=true)
        {
            var rigs = GameObject.FindObjectsOfType<Rig>();
            int i = 0;
            int j = 0;
            foreach (var rig in rigs)
            {
                rig.transform.root.transform.position = new Vector3(i * 3, 0, j * 3);
                ++i;
                if (i > 8)
                {
                    i = 0;
                    ++j;
                }
                
                rig.ik.SetAutoUpdate(activate);
            }
        }

        [Command]
        public static void testUnitSphereSpawn()
        {
            var rigs = GameObject.FindObjectsOfType<Rig>();
            var factory = GameObject.FindObjectOfType<DatasetFactory>();
            Debug.Log("# RIGS: " + rigs.Length);
            foreach (var rig in rigs)
            {
                var param = Planner.PointsInGrid(factory.degXRange, factory.degYRange,
                    factory.degZRange);
                var target = rig.targets[0];
             
                foreach (float3 p in param)
                {
                    var t = GameObject.Instantiate<GameObject>(target.gameObject, rig.transform.root);
                    Planner.TranslateTarget(t.GetComponent<Target>(), rig, p, 1);
                }
                if (rig.targets.Length > 1)
                {
                    var target2 = rig.targets[1];
             
                    foreach (float3 p in param)
                    {
                        
                        var t = GameObject.Instantiate<GameObject>(target2.gameObject, rig.transform.root);
                        Planner.TranslateTarget(t.GetComponent<Target>(), rig, p, -1 );
                    }
                }
            }
            
        }

        [Command]
        public static void LoadCLips()
        {
            var recorders = GameObject.FindObjectsOfType<Recorder>();
            var generator = GameObject.FindObjectOfType<SyntheticDataGenerator>();
            foreach (var recorder in recorders)
            {
                recorder.Load(generator);
            }
        }
        [Command]
        public static void LoadCLipsUsingPath()
        {
            var recorders = GameObject.FindObjectsOfType<Recorder>();
            var generator = GameObject.FindObjectOfType<SyntheticDataGenerator>();
            foreach (var recorder in recorders)
            {
                recorder.LoadUsingFilePath(generator);
            }
        }

        [Command]
        public static void ResetPosture()
        {
            var rigs = GameObject.FindObjectsOfType<Rig>();
            foreach (var rig in rigs)
            {
                rig.ResetPosture();
            }
        }

        [Command]
        public static void RecordSingleTarget(bool randomRot=false, float scaling=1)
        {
            var rigs = FindObjectsOfType<Rig>();
            foreach (var rig in rigs)
            {
                int id = UnityEngine.Random.Range(0, 39);
                rig.transform.localScale *= scaling;
                Recorder recorder = rig.GetComponent<Recorder>();
                recorder.Record(new []{id}, randomRot);
            }
        }

        [Command]
        public static void RecordMultipleTarget(int num = 2, bool randomRot=false, float scaling=1)
        {
            var rigs = FindObjectsOfType<Rig>();
            foreach (var rig in rigs)
            {
                rig.transform.localScale *= scaling;
                int[] id = new int[num];
                int k = 0;
                for (int i = 0; i < num; ++i)
                {
                    k = UnityEngine.Random.Range(0, 39);
                    foreach (var j in id)
                    {
                        if (k == j)
                        {
                            --i;
                            break;
                        }
                    }

                    id[i] = k;
                }
                Recorder recorder = rig.GetComponent<Recorder>();
                recorder.Record(id, randomRot);
            }
        }
        [Command]
        public static void RecordLoco(float amp=0.3f, float freq=0.3f, float stepSize=0.3f)
        {
            var rigs = FindObjectsOfType<Rig>();
            foreach (var rig in rigs)
            {
                
                Recorder recorder = rig.GetComponent<Recorder>();
                recorder.Record(amp, freq, stepSize);
            }
        }

        [Command]
        public static void RecordLocoTarget(float amp = 0.3f, float freq = 0.3f, float stepSize = 0.3f, int num = 2, bool randomRot=false)
        {
            var rigs = FindObjectsOfType<Rig>();
            foreach (var rig in rigs)
            {
                int[] id = new int[num];
                int k = 0;
                for (int i = 0; i < num; ++i)
                {
                    k = UnityEngine.Random.Range(0, 80);
                    foreach (var j in id)
                    {
                        if (k == j)
                        {
                            --i;
                            break;
                        }
                    }

                    id[i] = k;
                }
                Recorder recorder = rig.GetComponent<Recorder>();
                recorder.Record(id, randomRot, amp, freq, stepSize);
            }
        } 
        [Command]
        public static void save(string filename = "test")
        {
            Recorder[] recorders = FindObjectsOfType<Recorder>();
            SyntheticDataGenerator gen = FindObjectOfType<SyntheticDataGenerator>();
            foreach (var recorder in recorders)
            {
                gen.SaveSequence(recorder, filename, "Test");
            }
        }

        [Command]
        public static void printLevels()
        {
            var rigs = FindObjectsOfType<Rig>();
            foreach (var rig in rigs)
            {
                int numLevel0 = 0,
                    numLevel1 = 0,
                    numLevel2 = 0;

                foreach (var joint in rig.joints)
                {
                    if (joint.level == 0)
                        numLevel0++;
                    else if (joint.level == 1)
                        numLevel1++;
                    else if (joint.level == 2)
                        numLevel2++;
                }
                Debug.Log(rig.gameObject.name + ": " + numLevel0 + ", " + numLevel1 + ", " + numLevel2);
            }
        }
        [Command]
        public static void printNumKeyBones()
        {
            var rigs = FindObjectsOfType<Rig>();
            foreach (var rig in rigs)
            {
                int num = 0;

                foreach (var joint in rig.joints)
                {
                    if (joint.keyBone)
                        num++;

                }

                Debug.Log(rig.gameObject.name + ": " + num);
            }
        }


    }
}