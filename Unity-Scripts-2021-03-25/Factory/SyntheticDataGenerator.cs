using System;
using UnityEngine;
using Zenject;
using Anim;
using Tools;
using System.Collections.Generic;
using Unity.Mathematics;
namespace Factory
{
    public class SyntheticDataGenerator : MonoBehaviour
    {
        [Inject] private Planner _pathPlanner;
        [Inject(Id = "seed")] private int _seed = 0;

        private void Start()
        {
            if (_seed > 0){
                UnityEngine.Random.InitState(_seed);
            }
            else{
                UnityEngine.Random.InitState((int) Time.time);
            }
        }

        public void CreateSequence(Rig rig, Recorder recorder, float[] rTimings, Target.SpawnMode mode)
        {
          
            var path = _pathPlanner.GetRandomPath(rig);
            float[] timings = new float[rTimings.Length];
            float t = 0;
            for (int i = 0; i < rTimings.Length; ++i)
            {
                timings[i] = UnityEngine.Random.Range(t, rTimings[i]);
                t += rTimings[i];
            }

            foreach (var target in rig.targets)
            {
                target.RandomSpawn(mode);
            }

            recorder.Record(path, timings, _pathPlanner.GetRandomContactingFrames());
        }
        public void CreateSequence(Rig rig, Recorder recorder, float[] rTimings)
        {
          
            var path = _pathPlanner.GetRandomPath(rig);
            float[] timings = new float[rTimings.Length];
            float t = 0;
            for (int i = 0; i < rTimings.Length; ++i)
            {
                timings[i] = UnityEngine.Random.Range(t, rTimings[i]);
                t += rTimings[i];
            }


            recorder.Record(path, timings, _pathPlanner.GetRandomContactingFrames());
        }
        public void CreateSequenceUnitSphere(Rig rig, Recorder recorder, float4 spawnParam)
        {

            for (int i = 0, j=0; i < rig.targets.Length; ++i)
            {
                Vector3 targetPosition = Planner.TargetToRootSpace(rig.targets[i], rig, spawnParam);
                ++j;
                rig.targets[i].transform.position = targetPosition;
                rig.targets[i].transform.rotation =
                    Quaternion.AngleAxis(spawnParam.w, rig.targets[i].keyBone.transform.forward);
            }
            
            recorder.Record(_pathPlanner.GetRandomContactingFrames());
        }
        public void CreateSequenceGrid(Rig rig, Recorder recorder, float3 spawnParam, float rotation)
        {

            int reverse = 1;
            for (int i = 0; i < rig.targets.Length; ++i)
            {
                Planner.TranslateTarget(rig.targets[i], rig, spawnParam, reverse, rotation);
                reverse *= -1;
            }
            
            recorder.Record(_pathPlanner.GetRandomContactingFrames());
            
        }
        public void CreateSequence(Recorder recorder, float amplitude, float frequency, float stepSize)
        {
            recorder.Record(amplitude, frequency, stepSize); 
        }

        public void SaveSequence(Recorder recorder, string filename, string foldername)
        {
            var str = JsonUtility.ToJson(recorder.clip, true);
            DataIO.Save(str, filename, foldername);
        }
        public void LoadSequence(Recorder recorder, string filename, string foldername)
        {
            var rawJson = DataIO.Load(foldername+"/"+filename);
            var clip = Instantiate(recorder.prefabSequence);
            JsonUtility.FromJsonOverwrite(rawJson, clip);
            recorder.clip = clip;
        }

        public void LoadClips(Recorder recorder, string filename, string foldername)
        {
            var rawJson = DataIO.Load(filename);
            var clip = Instantiate(recorder.prefabSequence);
            var cc = JsonUtility.FromJson<Clip>(rawJson);
            
            clip.frames = new Anim.Data.Frame[cc.data.GetLength(0)];
            for (int i = 0; i < clip.frames.Length; ++i)
            {
                var f = new Anim.Data.Frame();
                f.joints = new Anim.Data.Joint[21];
                for (int j = 0; j < f.joints.Length; ++j)
                {
                    var jd = j * 3;
                    f.joints[j] = new Anim.Data.Joint();
                    f.joints[j].x.TargetValue = cc.data[i].joints[jd];
                    f.joints[j].y.TargetValue = cc.data[i].joints[jd+1];
                    f.joints[j].z.TargetValue = cc.data[i].joints[jd+2];
                }
                
                clip.frames[i] = f;

            }
            recorder.clip = clip; 
        }

        [Serializable]
        class Clip
        {
            
            public Jo[] data;
        }

        [Serializable]
        class Jo
        {
            public float[] joints;
        }
        
    }
}