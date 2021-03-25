using System;
using UnityEngine;
using Unity.Mathematics;
using BioIK;

namespace Anim {
    public class JointRecorder : MonoBehaviour {
        /*
         * This component is attached to every joint and contains joint-specific information that are logged.
         */
        public BioJoint joint;
        [Header("Target info")]
        public bool keyBone = false;
        public int isLeft = 0;
        public int chainPos = 0;
        public Transform target;
        [SerializeField] private bool contact = false;
        
        [Header("Debug UI")]
        public bool trajectoryBone = false;
        
        [Header("Generation")]
        public bool randomBone = false;
        public float threshold = 0.1f;
        
        [Header("Physics info")]
        public Vector3 centerOfMass;
        public float totalMass;
        public float3x3 inertiaObj;
        public Vector3 length;
        public float radius;

        [Header("Initial attributes")]
        private float3 initPosition;
        private quaternion initRotation;
        private float3 initRotationEuler;
        private float3x3 initOrientation;
        
        private Transform _root;
        private void Awake()
        {
            joint = GetComponent<BioJoint>();
            _root = transform.root;
            threshold = 0.1f;
            
            /*
             * Label the joints based one their names.
             * isLeft = [-1,0,1]    
             * chainPos = [-1,0,1]  , shoulder and thighs are assigned to 1, hands and feet are assigned to -1, rest is 0
             */
            if (gameObject.name.Contains("left") ||
                gameObject.name.Contains("Left") ||
                gameObject.name.Contains("_l"))
                isLeft = 1;
            else if (gameObject.name.Contains("right") ||
                gameObject.name.Contains("Right") ||
                gameObject.name.Contains("_r"))
                isLeft = -1;
            else
            {
                isLeft = 0;
            }
            
            if (gameObject.name.Contains("shoulder") ||
                gameObject.name.Contains("Shoulder") ||
                gameObject.name.Contains("Thigh") ||
                gameObject.name.Contains("thigh") ||
                gameObject.name.Contains("UpLeg")) 
                chainPos = 1;
            else if (gameObject.name.Contains("hand") ||
                     gameObject.name.Contains("Hand") ||
                     gameObject.name.Contains("Wrist") ||
                     gameObject.name.Contains("Ankle") ||
                     gameObject.name.Contains("Foot") ||
                     gameObject.name.Contains("foot"))
                chainPos = -1;
            else
            {
                chainPos = 0;
            }
            
            centerOfMass = _root.InverseTransformPoint(transform.position);
            totalMass = 1000 * math.PI * math.pow(radius, 2) * length.magnitude;
            inertiaObj = new float3x3(
                new float3(ReturnInertia(0), 0, 0),
                new float3(0, ReturnInertia(1), 0),
                new float3(0, 0, ReturnInertia(2))
            );

            initPosition =_root.InverseTransformPoint(transform.position); 
            initRotation =_root.transform.rotation * transform.rotation; 
            initRotationEuler =_root.InverseTransformDirection(transform.rotation.eulerAngles); 
            initOrientation = new float3x3(
                _root.InverseTransformDirection(transform.forward),
                _root.InverseTransformDirection(transform.up),
                _root.InverseTransformDirection(transform.right));


        }

        public void SetJointData(Data.Joint jData, Data.Joint prev=null)
        {
            jData.isLeft = isLeft;
            jData.chainPos = chainPos;
            
            jData.position = _root.InverseTransformPoint(transform.position);
            jData.rotEuler = _root.InverseTransformDirection(transform.rotation.eulerAngles);
            jData.rotQuaternion = _root.transform.rotation * transform.rotation;
            jData.rotMat = new float3x3(
                _root.InverseTransformDirection(transform.forward),
                _root.InverseTransformDirection(transform.up),
                _root.InverseTransformDirection(transform.right));

            jData.geoDistance = joint.geoDistance; 
            jData.geoDistanceNormalised = joint.geoDistanceNormalised; 
            
            jData.inertiaObj = inertiaObj;
            // jData.COM = centerOfMass;
            jData.totalMass = totalMass;
            if (prev == null)
            {
                jData.velocity = (jData.position - initPosition) / Time.fixedDeltaTime;
                jData.angularVelocity = (jData.rotEuler - initRotationEuler) / Time.fixedDeltaTime;
            }
            else
            {
                jData.velocity = (jData.position - prev.position) / Time.fixedDeltaTime;
                jData.angularVelocity = (jData.rotEuler - prev.rotEuler) / Time.fixedDeltaTime;
                
            }
            jData.linearMomentum = jData.velocity * totalMass;
            jData.inertia = math.mul(jData.rotMat, math.mul(jData.inertiaObj, math.transpose(jData.rotMat)));
            jData.angularMomentum = math.mul(jData.inertia, jData.angularVelocity);
            
            jData.x = new BioJoint.Motion(joint.X);
            jData.y = new BioJoint.Motion(joint.Y);
            jData.z = new BioJoint.Motion(joint.Z);

            if (keyBone)
            {
                jData.tGoalPosition = target.InverseTransformPoint(transform.position);
                jData.tGoalDirection = target.InverseTransformDirection(transform.rotation.eulerAngles);

                jData.cost = DistanceObjective.ComputeCost(transform, target, _root);
                
                jData.contact = Vector3.Distance(target.transform.position, transform.position) < threshold;
                contact = jData.contact;
                jData.key = keyBone;

            }
        }

        public bool HasRoot(){return _root != null;}

        private float ReturnInertia(int i)
        {
            /*
             * Assumes the joints have the inertia of uniform cylinder.
             */
            float h = length.magnitude;
            if (i == 0 || i == 1)
                return (1f / 12f) * totalMass * math.pow(h, 2) + (1f / 4f) * totalMass * math.pow(radius, 2);
            if (i == 2)
                return (1f / 2f) * totalMass * math.pow(radius, 2);
            else
            {
                return 0;
            }
        }

        public void Reset()
        {
            joint.Reset();
        }

    }
}