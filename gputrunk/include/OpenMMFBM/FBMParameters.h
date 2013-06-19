#ifndef FBM_PARAMETERS_H_
#define FBM_PARAMETERS_H_

#include <vector>
#include <string>

namespace OpenMMFBM {
    namespace Preference {
        enum EPlatform { Reference, OpenCL, CUDA };
    }
    
    struct Force {
        Force( std::string n, int i ) : name( n ), index( i ) {}
        std::string name;
        int index;
    };

    struct FBMParameters {
        double blockDelta;
        double sDelta;
        std::vector<int> residue_sizes;
        int res_per_block;
        int bdof;
        std::vector<Force> forces;
        int modes;
        
        Preference::EPlatform blockPlatform;
        
        FBMParameters();
    };
}
    
#endif // FBM_PARAMETERS_H_
