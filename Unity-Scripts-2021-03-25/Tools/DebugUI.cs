using UnityEngine;
using Anim;
using Drawing;
using Zenject;

namespace Tools
{
    public class DebugUI : MonoBehaviour
    {
        [Inject] private Rig _rig;
        [Inject] private MainSystem _sys;

        // private void Update()
        // {
        //     if (_sys.displayAnchor)
        //     {
        //         foreach (var j in _rig.joints)
        //         {
        //             using (Draw.InLocalSpace(j.transform))
        //             {
        //                 Draw.Label2D(j.transform.up * 0.1f, j.joint.GetAnchorInWorldSpace().ToString("F3"), 7f);
        //             }
        //         }
        //
        //     }
        //
        //     if (_sys.displayOrientation)
        //     {
        //         foreach (var j in _rig.joints)
        //         {
        //             using (Draw.InLocalSpace(j.transform))
        //             {
        //                 Draw.Label2D(j.transform.up * 0.15f, j.joint.GetOrientation().ToString("F3"), 7f);
        //             }
        //         }
        //     }
        // }
    }
}