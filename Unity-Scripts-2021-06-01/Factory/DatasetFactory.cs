using UnityEngine;
using Anim;
using System.Collections;
using System.Collections.Generic;
// using MiscUtil.Collections.Extensions;
using Zenject;
using Unity.Mathematics;
namespace Factory
{
    public class DatasetFactory : MonoBehaviour
    {
        [Header("Dataset name")] public string datasetName = "Test";
        
        [Header("Dataset parameters")]
        public int samples = 1000;
        public Target.SpawnMode spawnMode = Target.SpawnMode.InsideSphere;
        
        [Header("Generation parameters")]
        public int agents = 10;
        public GameObject[] characterPrefabs;
        public int intermediatePoses = 2;
        public float[] rTimings;

        public float rotation;
        public Vector3 degXRange;
        public Vector3 degYRange;
        public Vector3 degZRange;
        
        public Vector3 amplitudeRange; // (min, max, stepSize)
        public Vector3 frequencyRange;
        public Vector3 stepSizeRange;

        public int[] num_targets_range;
        
        [Inject] private Planner _pathPlanner;
        [Inject] private SyntheticDataGenerator _sequenceMaker;
        private Rig[,] _agentInstances;
        private Recorder[,] _agentRecorders;
        public void GenerateBasic()
        {
            SpawnAgents();
            int batchSize = samples / agents;
            for (int i = 0; i < agents; ++i)
            {
                StartCoroutine(MakeData(i, batchSize));
            }
             
        }
        public void GenerateSystematic(bool ifRotate=false)
        {
            SpawnAgents();
            var spawnParam = Planner.PointsInGrid(degXRange, degYRange, degZRange);
            samples = spawnParam.Length * characterPrefabs.Length;
            Debug.Log("Samples: " +spawnParam.Length);
            int batchSize = samples / agents;
            float rotation = (ifRotate) ? this.rotation : 0;
            // var spawnParam = Planner.PointsInUnitSphere(rRange, degXRange, degYRange, degZRange);
            for (int i = 0; i < agents; ++i)
            {
                StartCoroutine(MakeDataBatch(i, batchSize, spawnParam, rotation));
            }
             
        }
        
        public void GenerateVersion2()
        {
            SpawnAgents();

            var param = new List<float2>();
            for (int i = 0; i < 2; i++)
            {
                foreach (int n_targets in num_targets_range)
                {
                    float2 p = new float2(i, n_targets);
                    param.Add(p);
                }
            }
            
            samples = param.Count * characterPrefabs.Length;
            Debug.Log("Samples: " + samples);
            
            int batchSize = samples / agents;
            for (int i = 0; i < agents; ++i)
            {
                StartCoroutine(MakeDataBatch(i, batchSize, param.ToArray()));
            }
             
        }
        public void GenerateLocomotion()
        {
            var nAmp = (int) ((amplitudeRange.y - amplitudeRange.x) / amplitudeRange.z);
            var nFreq = (int) ((frequencyRange.y - frequencyRange.x) / frequencyRange.z);
            var nStep = (int) ((stepSizeRange.y - stepSizeRange.x) / stepSizeRange.z);

            samples = nAmp * nFreq * nStep;
            Debug.Log(nAmp + " " + nFreq + " " + nStep);
            Debug.Log("Samples: " + samples);
            SpawnAgents();
            int batchSize = samples / agents;
            List<Vector3> args = new List<Vector3>();
            int id = 0;
            for (int i = 0; i < nAmp; ++i)
            {
                for (int j = 0; j < nFreq; ++j)
                {
                    for (int k = 0; k < nStep; ++k)
                    {
                        args.Add(new Vector3(amplitudeRange.x + amplitudeRange.z * i,
                            frequencyRange.x + frequencyRange.z * j,
                            stepSizeRange.x + stepSizeRange.z * k));
                        if (args.Count >= batchSize)
                        {
                            StartCoroutine(MakeData(id, batchSize, args));
                            args = new List<Vector3>();
                            ++id;
                        }
                    }
                }
            }
        }

        private IEnumerator MakeData(int id, int batchSize)
        {
            yield return new WaitForFixedUpdate();
            for (int i = 0; i < batchSize; ++i)
            {
                for (int j = 0; j < characterPrefabs.Length; ++j)
                {
                    var rig = _agentInstances[id,j];
                    var recorder = _agentRecorders[id,j];
                    _sequenceMaker.CreateSequence(rig, recorder, rTimings, spawnMode);             
                    _sequenceMaker.SaveSequence(recorder, (id*batchSize+i).ToString(), datasetName+"_"+j);
                }
            }
        }
        private IEnumerator MakeData(int id, int batchSize, List<Vector3> args)
        {
            yield return new WaitForFixedUpdate();
            int j = 0;
            foreach (var arg in args)
            {
                for (int i = 0; i < characterPrefabs.Length; ++i)
                {
                    var recorder = _agentRecorders[id,i];
                    
                    _sequenceMaker.CreateSequence(recorder, arg.x, arg.y, arg.z);             
                    _sequenceMaker.SaveSequence(recorder, (id*batchSize+j).ToString(), datasetName+"_"+characterPrefabs[i].name);
                }
                ++j;
            }
        }
        private IEnumerator MakeDataBatch(int id, int batchSize, float3[] spawnParam, float rotation)
        {
            yield return new WaitForFixedUpdate();
            float3 param;
            int n = 0;
            for (int i = 0; i < batchSize; i++)
            {
                n = id * batchSize + i;
                if (n >= spawnParam.Length){
                    break;
                }

                param = spawnParam[n];
                for (int j = 0; j < characterPrefabs.Length; ++j)
                {
                    var rig = _agentInstances[id,j];
                    var recorder = _agentRecorders[id,j];
                    _sequenceMaker.CreateSequenceGrid(rig, recorder, param, rotation);             
                    _sequenceMaker.SaveSequence(recorder, (n).ToString(), datasetName+"_"+characterPrefabs[j].name);
                }
            }
        }
        private IEnumerator MakeDataBatch(int id, int batchSize, float2[] spawnParam)
        {
            yield return new WaitForFixedUpdate();
            float2 param;
            int n = 0;
            for (int i = 0; i < batchSize; i++)
            {
                n = id * batchSize + i;
                if (n >= spawnParam.Length){
                    break;
                }

                param = spawnParam[n];
                bool randomRot = (int) param.x == 1;
                int num_targets = (int) param.y;

                string name = $"{randomRot}_{num_targets}_";
                for (int j = 0; j < characterPrefabs.Length; ++j)
                {
                    var rig = _agentInstances[id,j];
                    var recorder = _agentRecorders[id,j];
                    for (int k = 0; k < 40; ++k)
                    {
                        _sequenceMaker.CreateSequenceVersion2(rig, recorder, randomRot, num_targets, k);             
                        _sequenceMaker.SaveSequence(recorder, name+k, datasetName+"_"+characterPrefabs[j].name);
                    }
                }
            }
        }

        private void SpawnAgents(){
            _agentInstances = new Rig[agents,characterPrefabs.Length];
            _agentRecorders = new Recorder[agents, characterPrefabs.Length];

            for (int i = 0; i < agents; ++i){
                for (int j = 0; j < characterPrefabs.Length; ++j)
                {
                    var instance = Instantiate<GameObject>(characterPrefabs[j]);
                    _agentInstances[i,j] = instance.GetComponentInChildren<Rig>();
                    _agentRecorders[i,j] = instance.GetComponentInChildren<Recorder>();
                    
                }
            }

        }
    }
}