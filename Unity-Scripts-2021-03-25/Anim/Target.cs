using System;
using UnityEngine;
using Zenject;
using Drawing;

namespace Anim
{
    public class Target: MonoBehaviour
    {
        public Vector3 spawnCenter;
        public float spawnRadius;
        public float spawnRadiusXL;
        public float interactionRadius = 0.1f;
        public Transform keyBone;
        public Transform pelvis;
        public BioIK.BioJoint joint;
        [Inject(Id = "seed")] private int _seed = 0;
        
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
            joint = joRecorder.joint;
            joRecorder.target = transform;
            
            pelvis = keyBone.transform.root;
            
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
                    RandomSpawnOnSphere();
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

       
    }
}