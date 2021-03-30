using System;
using System.Collections;
using UnityEngine;
using Zenject;
using Anim;
namespace Factory
{
    public class Locomotion : MonoBehaviour
    {
        public Rig _rig;
        public Transform targetL, targetR;
        
        [SerializeField] private float phaseL, phaseR;
        [SerializeField] private float amplitude;
        [SerializeField] private float freq;
        [SerializeField] private float t = 0;
        [SerializeField] private float stepSize = 0;
        [SerializeField] private float noiseStrength = 0;
        [SerializeField] private bool rootMotion = false;

        private float rx, lx,rz, lz, rigY;

        private Transform root;

        private Vector3 defaultPosR, defaultPosL, defaultPosRig;
        private Vector3 groundPos;
        private void Start()
        {
            root = transform.root;
            defaultPosL = targetL.transform.position;
            defaultPosR = targetR.transform.position;
            rx = defaultPosR.x;
            rz = defaultPosR.z;
            lx = defaultPosL.x;
            lz = defaultPosL.z;

            defaultPosRig = _rig.transform.position;
            rigY = defaultPosRig.y;
            groundPos = (defaultPosL + defaultPosR) / 2f;

        }

        public void Step()
        {
            var noise = Planner.Normal(0, 1)* noiseStrength; 
            var noise2 = Planner.Normal(0, 1) * noiseStrength; 
            
            var sl = Sinusoidal(phaseL);
            var sr = Sinusoidal(phaseR);
            lz += Mathf.Max(-0.01f, stepSize * sl * root.localScale.x); 
            rz += Mathf.Max(-0.01f, stepSize * sr * root.localScale.x); 
            targetL.transform.position = new Vector3(lx + noise, Mathf.Max(0,sl), lz) + groundPos;
            targetR.transform.position = new Vector3(rx + noise2, Mathf.Max(0,sr), rz) + groundPos;
            _rig.transform.position = new Vector3((lx + rx) / 2f, rigY, (lz + rz) / 2f);
            ++t;
        }

        private float Sinusoidal(float p)
        {
            
            return amplitude * Mathf.Sin(p + t * freq);
        }
        
        IEnumerator WaitForStep(float t, float duration)
        {
            _rig.ik.SetAutoUpdate(true);
            while (duration > 0)
            {
                Step();
                yield return new WaitForSeconds(t);
                duration -= t;
            }
            _rig.ik.SetAutoUpdate(false);
        }

        public void Walk(float t, float duration)
        {
            StartCoroutine(WaitForStep(t, duration));
        }

        public void Reset()
        {
            targetL.transform.position = defaultPosL;
            targetR.transform.position = defaultPosR;
            _rig.transform.position = defaultPosRig;
            t = 0;
            rz =  lz = 0;
        }

        public void SetStepSize(float s)
        {
            stepSize = s;
        }

        public void SetFrequency(float f)
        {
            freq = f;
        }

        public void SetAmplitude(float a)
        {
            amplitude = a;
        }

    }
}