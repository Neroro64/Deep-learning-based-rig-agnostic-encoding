using UnityEngine;
using QFSW.QC;
using Anim;
using Factory;

namespace Tools
{
    public class Commands : MonoBehaviour
    {
        [Command]
        public static void CreateDatasetBasic(string datasetName)
        {
            var factory = FindObjectOfType<DatasetFactory>();
            factory.datasetName = datasetName;
            
            factory.GenerateBasic();
        }
        [Command]
        public static void CreateDatasetLocomotion(string datasetName)
        {
            var factory = FindObjectOfType<DatasetFactory>();
            factory.datasetName = datasetName;
            
            factory.GenerateLocomotion();
        }
        [Command]
        public static void CreateDatasetSystematic(string datasetName, bool ifRotate=false)
        {
            var factory = FindObjectOfType<DatasetFactory>();
            factory.datasetName = datasetName;
            
            factory.GenerateSystematic(ifRotate);
        } 
        [Command]
        public static void CreateDatasetVersion2(string datasetName)
        {
            var factory = FindObjectOfType<DatasetFactory>();
            factory.datasetName = datasetName;
            
            factory.GenerateVersion2();
        } 

    }
}