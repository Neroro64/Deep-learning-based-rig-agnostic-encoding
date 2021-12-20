using Anim;
using UnityEngine;
using Zenject;
using Factory;

namespace Tools
{
    public class DependencyInstaller : MonoInstaller
    {
        public Anim.Rig rig;
        public MainSystem sys;
        public Recorder recorder;
        public Planner planner;
        public SyntheticDataGenerator dataGenerator;
        public int seed = 2021;
        public override void InstallBindings()
        {
            Container.Bind<MainSystem>().FromInstance(sys).NonLazy();
            Container.Bind<Anim.Rig>().FromInstance(rig).NonLazy();
            Container.Bind<Anim.Recorder>().FromInstance(recorder).NonLazy();
            Container.Bind<Factory.Planner>().FromInstance(planner).NonLazy();
            Container.Bind<Factory.SyntheticDataGenerator>().FromInstance(dataGenerator).NonLazy();
            Container.Bind<int>().WithId("seed").FromInstance(seed).NonLazy();
            
            
        }
    }
}