using System;
using UnityEngine;
using Anim;
using System.Collections.Generic;
using System.IO.Pipes;
using Unity.Mathematics;
using Zenject;
using Random = System.Random;

namespace Factory {
    public class Planner : MonoBehaviour {
        [SerializeField] private int keyframes = 2;
        [SerializeField] private float2 contactingFramesParam = new float2(5, 2);
        // [SerializeField] private float[] timing;
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

        public List<float3[]> GetRandomPath(Rig rig){
            var path = new List<float3[]>();

            var poseL = new List<float3>();
            var poseX = new List<float2>();
            var poseY = new List<float2>();
            var poseZ = new List<float2>();

            foreach(var j in rig.joints){
                if (j.keyBone){
                    poseL.Add(new float3());
                    poseX.Add(new float2((float)j.joint.X.LowerLimit, (float)j.joint.X.UpperLimit));
                    poseY.Add(new float2((float)j.joint.Y.LowerLimit, (float)j.joint.Y.UpperLimit));
                    poseZ.Add(new float2((float)j.joint.Z.LowerLimit, (float)j.joint.Z.UpperLimit));
                }
            }

            var xLim = poseX.ToArray();
            var yLim = poseY.ToArray();
            var zLim = poseZ.ToArray();
            for (int i = 0; i < keyframes; ++i){
                for (int j = 0; j < poseL.Count; ++j){
                    poseL[j] = new float3(
                        UnityEngine.Random.Range(xLim[j].x, xLim[j].y),
                        UnityEngine.Random.Range(yLim[j].x, yLim[j].y),
                        UnityEngine.Random.Range(zLim[j].x, zLim[j].y)
                        );
                }       

                path.Add(poseL.ToArray());
            }
            return path;
        }

        public int GetRandomContactingFrames()
        {
            // return (int)Normal(contactingFramesParam[0], contactingFramesParam[1]);
            return (int)contactingFramesParam[0];
        }
        
        public static float Normal(float mu, float sigma)
        {
            float rand1 = UnityEngine.Random.Range(0.0f, 1.0f);
            float rand2 = UnityEngine.Random.Range(0.0f, 1.0f);

            float n = Mathf.Sqrt(-2.0f * Mathf.Log(rand1)) * Mathf.Cos((2.0f * Mathf.PI) * rand2);

            return (mu + sigma * n);
        }

        public static List<float4> RandomPointInUnitSphere(int n=1)
        {
            List<float4> param = new List<float4>();
            
            for(int i= 0; i < n; ++i)
            {
                param.Add(new float4(
                    UnityEngine.Random.Range(0.2f, 1f), // r 
                    UnityEngine.Random.Range(-90, 90), // degX
                    UnityEngine.Random.Range(-90, 90), // degY
                    UnityEngine.Random.Range(-180, 180) // degZ
                ));
            }

            return param;
        }
        public static float4[] PointsInUnitSphere(float3 rRange, float3 degXRange, float3 degYRange, float3 degZRange)
        {
            int degXRangeLen = (int) ((degXRange.y - degXRange.x) / degXRange.z);
            int degYRangeLen = (int ) ((degYRange.y - degYRange.x) / degYRange.z);
            int degZRangeLen = (int) ((degZRange.y - degZRange.x) / degZRange.z);
            int rRangeLen = (int) ((rRange.y - rRange.x) / rRange.z);

            int n = degXRangeLen * degYRangeLen * degZRangeLen * rRangeLen;

            var param = new float4[n];
            int k = 0;
            for (int z = 0; z < degZRangeLen; ++z)
            {
                for (int r = 0; r < rRangeLen; ++r)
                {
                    for (int y = 0; y < degYRangeLen; ++y)
                    {
                        for (int x = 0; x < degXRangeLen; ++x)
                        {
                            param[k] = new float4(r, x, y, z);
                            ++k;
                        }
                    }
                }
                
            }
           
            return param;
        }
        public static float3[] PointsInGrid(float3 xRange, float3 yRange, float3 zRange) 
        {
            int degXRangeLen = (int) ((xRange.y - xRange.x) / xRange.z);
            int degYRangeLen = (int ) ((yRange.y - yRange.x) / yRange.z);
            int degZRangeLen = (int) ((zRange.y - zRange.x) / zRange.z);

            int n = degXRangeLen * degYRangeLen * degZRangeLen;

            var param = new float3[n];
            int k = 0;
            for (int z = 0; z < degZRangeLen; ++z)
            {
                for (int y = 0; y < degYRangeLen; ++y)
                {
                    for (int x = 0; x < degXRangeLen; ++x)
                    {
                        param[k] = new float3(x, y, z);
                        ++k;
                    }
                }
            }
           
            return param;
        }
        public static Vector3 TargetToRootSpace(Target t, Rig rig, float4 param)
        {
            float radius = param.x,
                  degX = param.y,
                  degY = param.z,
                  degZ = param.w;
            
            float r = radius * Vector3.Distance(t.keyBone.transform.position ,rig.transform.position);
            Vector3 dir = Vector3.Normalize(
                new Vector3(Mathf.Cos(Mathf.Deg2Rad * degX), Mathf.Sin(Mathf.Deg2Rad * degX), r));
            dir *= r;
            return t.pelvis.transform.position + dir;
        }
        
        public static void TranslateTarget(Target t, Rig rig, float3 param, int reverse, float rotation=0)
        {
            var pelvis = t.pelvis.transform;
            var jo = t.joint; 
            
            // Vector3 posX = pelvis.right * param.x;
            // Vector3 posY = -pelvis.up * param.y;
            // Vector3 posZ = pelvis.forward * param.z;

            Vector3 posX = -Vector3.right * ( (param.x - 0.5f) * reverse);
            Vector3 posY = Vector3.up * param.y;
            Vector3 posZ = Vector3.forward * (param.z + 1f);
        
                
            // Vector3 pos = Vector3.Normalize(posX + posY + posZ) * Vector3.Distance(t.keyBone.transform.position, rig.transform.position);
            Vector3 pos = Vector3.Normalize(posX + posY + posZ);
            pos *= Vector3.Distance(jo.transform.position, jo.Segment.Parent.transform.position) * 2f;
            
            if (rotation > 0)
                t.transform.Rotate(t.transform.right, rotation, Space.World);
       
            t.transform.position = pelvis.position + pos;
            
        }
        
    }
}