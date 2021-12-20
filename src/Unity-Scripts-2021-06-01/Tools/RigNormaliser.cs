using System;
using UnityEngine;
using Anim;

namespace Tools
{
    public class RigNormaliser : MonoBehaviour
    {
        public Rig rig;
        public GameObject prefab;
        public bool normalised = false;
        public float len = 1f;

        private GameObject[] _instances;
        private JointRecorder[] _joints;
        private void Start()
        {
            GameObject ga;
            Vector3 vec;
            float f;
            _instances = new GameObject[rig.joints.Length];
            _joints = new JointRecorder[rig.joints.Length];
            
            int i = 0;
            foreach (var jo in rig.joints)
            {
                ga = Instantiate<GameObject>(prefab, transform);
                vec = Vector3.Normalize(jo.transform.position - jo.transform.root.position);
                if (normalised)
                    f = jo.joint.geoDistanceNormalised;
                else
                    f = jo.joint.geoDistance;
                ga.transform.localPosition = vec * f;
                ga.transform.rotation = jo.transform.rotation;
                _instances[i] = ga;
                _joints[i] = jo;
                ++i;
            }
        }

        private void LateUpdate()
        {
            JointRecorder jo;
            GameObject ga;
            Vector3 vec;
            float f;
            for (int i = 0; i < _joints.Length; ++i)
            {
                jo = _joints[i];
                ga = _instances[i];
                vec = Vector3.Normalize(jo.transform.position - jo.transform.root.position);
                if (normalised)
                    f = jo.joint.geoDistanceNormalised;
                else
                    f = jo.joint.geoDistance;
                ga.transform.localPosition = vec * ( f * len );
                ga.transform.rotation = jo.transform.rotation; 
            }
        }
    }
}