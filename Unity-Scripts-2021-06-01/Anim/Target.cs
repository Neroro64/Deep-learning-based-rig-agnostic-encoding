using System;
using UnityEngine;
using Zenject;
using Drawing;
using Unity.Mathematics;
using UnityEngine.SocialPlatforms;
using DG.Tweening;

namespace Anim
{
    public class Target: MonoBehaviour
    {
        public Vector3 spawnCenter;
        public float spawnRadius;
        public float spawnRadiusXL;
        public float interactionRadius = 0.2f;
        public int num = 40;
        public GameObject testPrefab;
        public float offset = 0;
        public bool isLoco = false;
        
        public Vector3 initPosition;
        public Quaternion initRotation;
        public float3x3 initRotMat;
        
        public Transform keyBone;
        public Transform pelvis;
        public JointRecorder joint;
        [Inject(Id = "seed")] private int _seed = 0;
        
        
        public Vector3[] positions;
        private bool spawned = false;
        private int posID = 0;
        public bool reverse = false;

        private Sequence seqP, seqR;
        public float duration = 1f;
        public enum SpawnMode
        {
            InsideSphere, OutsideSphere, OnPlane, OnSphere
        }


        private void Start()
        {
            if (_seed > 0)
                UnityEngine.Random.InitState(_seed);
            else
                UnityEngine.Random.InitState((int)Time.time);

            var joRecorder = keyBone.GetComponent<JointRecorder>(); 
            joint = joRecorder;
            joRecorder.target = transform;
            joRecorder.Target = this;
            
            // pelvis = keyBone.transform.root;
            
            initPosition = transform.position;
            initRotation = transform.rotation;
            initRotMat = new float3x3(transform.forward, transform.up, transform.right);

            seqP = DOTween.Sequence();
            seqR = DOTween.Sequence();
            if (!isLoco)
                GeneratePositions();
        }

        public void RandomSpawn(SpawnMode mode = SpawnMode.InsideSphere)
        {            
         
            switch (mode)
            {
                default:
                case SpawnMode.InsideSphere:
                    RandomSpawnInsideSphere();
                    break;
                case SpawnMode.OutsideSphere:
                    RandomSpawnInsideSphere(false);
                    break;
                case SpawnMode.OnPlane:
                    RandomSpawnOnPlane();
                    break;
                case SpawnMode.OnSphere:
                    // RandomSpawnOnSphere();
                    SpawnOnSphere();
                    break;
            }
        } 
        private void RandomSpawnInsideSphere(bool reachable = true)
        {
            if (reachable)
                transform.position = UnityEngine.Random.insideUnitSphere * spawnRadius + spawnCenter ;
            else
                transform.position = UnityEngine.Random.insideUnitSphere * spawnRadiusXL + spawnCenter ;
            transform.rotation = UnityEngine.Random.rotation;
        }
        private void RandomSpawnOnSphere()
        {
            transform.position = UnityEngine.Random.onUnitSphere * spawnRadius + spawnCenter ;
            transform.rotation = UnityEngine.Random.rotation;
        }
        private void RandomSpawnOnPlane()
        {
            transform.position = new Vector3(
                UnityEngine.Random.Range(spawnCenter.x - spawnRadius, spawnCenter.x + spawnRadius),
                transform.position.y,
                UnityEngine.Random.Range(spawnCenter.z - spawnRadius, spawnCenter.z + spawnRadius));
            transform.rotation = UnityEngine.Random.rotation;
        }

        private void SpawnOnSphere()
        {
            float rx, ry, rz;
            rx = pelvis.transform.position.x;
            ry = pelvis.transform.position.y;
            rz = pelvis.transform.position.z;
            
            float dr = spawnRadius / 5f;
            int n = num / 5;
            positions = new Vector3[num];
            int k = 0; 
            for (int i = 1; i <= 5; i++)
            {
                float y = -spawnRadius + i * 2 * dr;
                float ds = 2 * math.PI / n;
                
                for (int j = 0; j < n; j++)
                {
                    var instance = Instantiate<GameObject>(testPrefab);
                    instance.transform.position =
                        new Vector3(spawnRadius * math.cos(ds * j) + rx, y+ry, spawnRadius * math.sin(ds * j)+rz);
                    positions[k++] = instance.transform.position;
                }
            }

            if (reverse)
                posID = n - 1;
            
            spawned = true;
        }


        public void GeneratePositions()
        {
            float dr = spawnRadius / 5f;
            int n = num / 5;
            positions = new Vector3[num];
            int k = 0;
            Vector3 pos;
            for (int i = 1; i <= 5; i++)
            {
                float y = -spawnRadius + i * 2 * dr;
                float ds = 2 * math.PI / n;
                int m = 0;
                for (int j = 0; j < n; j++)
                {
                    
                    if (reverse)
                        m = n - j - 1;
                    else
                        m = j;
                    pos = new Vector3(spawnRadius * math.cos(ds * m + offset), y, spawnRadius * math.sin(ds * m + offset));

                    positions[k++] = pos;
                }
            }
        }

        public void SetPosition(int id, bool randomRotation=false)
        {
            if (!isLoco)
            {
                if (id == -1)
                {
                    transform.position = initPosition;
                    transform.rotation = initRotation;
                }
                else
                    transform.position = (positions[id] + pelvis.position) * pelvis.localScale.x;

                if (randomRotation)
                    transform.rotation = UnityEngine.Random.rotation;
            }
        }
        
        public void SetPositionI(int id, bool randomRotation=false)
        {
            if (!isLoco)
            {
                if (id == -1)
                {
                    if (!seqP.IsPlaying())
                        seqP.Append(transform.DOMove(initPosition, duration));
                    if (!seqR.IsPlaying())
                        seqR.Append(transform.DORotateQuaternion(initRotation, duration));
                    
                }
                else
                    if (!seqP.IsPlaying())
                        seqP.Append(transform.DOMove(positions[id] + pelvis.transform.position, duration));

                if (randomRotation)
                    if (!seqR.IsPlaying())
                        seqR.Append(transform.DORotateQuaternion(UnityEngine.Random.rotation, duration));
            }
        }

        
        

       
    }
}