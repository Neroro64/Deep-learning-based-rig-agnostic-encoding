using System;
using UnityEngine;
using System.Linq;
using System.IO;
namespace Tools
{
    public class DataIO
    {
        static string _path = Application.persistentDataPath + "/";
        public static void Save(string data, string filename)
        {
            try
            {
                File.WriteAllText(_path + filename + ".json", data);
            }
            catch { 
                Debug.LogWarning("Save failed"); 
                }
        }
        public static void Save(string data, string filename, string folder)
        {
            try
            {
                bool existsFolder = Directory.Exists(_path + folder);
                if (!existsFolder){
                    Directory.CreateDirectory(_path+folder);
                }

                File.WriteAllText(_path + folder+ "/" + filename + ".json", data);
            }
            catch{
                Debug.LogWarning("Save failed"); 
            }
        }

        public static string Load(string filename, bool prepend=true)
        {
            try
            {
                string data;
                if (prepend)
                {
                    data = File.ReadAllText(_path + filename + ".json");
                }
                else
                {
                    data = File.ReadAllText(filename);
                }
                return data;
            }
            catch { 
                Debug.LogWarning("Load failed");
            }
            return null;
        }
        public static void Delete(string filename)
        {
            try
            {
                string[] matches = Directory.GetFiles(_path, filename + "*");
                foreach (var file in matches)
                {
                    File.Delete(file);
                }
            }
            catch
            {
                Debug.LogWarning("Delete file failed");
            }
        }
        public static void DeleteDir(string dir)
        {
            try
            { 
                if (Directory.Exists(_path + dir))
                    Directory.Delete(_path + dir, true);
            }
            catch {
                Debug.LogWarning("Delete directory failed"); 
            }
        }
    }
}