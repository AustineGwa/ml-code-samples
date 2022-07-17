package ai;

import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;

public class NDArray {

    public static void main(String[] args) {

        /*
        NDManager
        helps manage the memory usage of the NDArrays.It creates them and helps clear them as well.
        Once you finish using an NDManager, it will clear all of the NDArrays that were created within it’s scope as well.
        NDManager helps the overall system utilize memory efficiently by tracking the NDArray usage.

         */

//        try(NDManager manager = NDManager.newBaseManager()) {
////            NDArray Creation
////            ones is an operation to generate N-dim array filled with 1.
//            NDArray nd = (NDArray) manager.ones(new Shape(2, 3));
//                /*
//                ND: (2, 3) cpu() float32
//                [[1., 1., 1.],
//                 [1., 1., 1.],
//                ]
//                */
//
////            You can also try out random generation.
////            For example, we will generate random uniform data from 0 to 1.
//            NDArray nd2 = (NDArray) manager.randomUniform(0, 1, new Shape(1, 1, 4));
//
////            We can also try some math operations using NDArrays.
////            Assume we are trying to do a transpose and add a number to each element of the NDArray. We can achieve this by doing the following:
//
//            NDArray nd3 = (NDArray) manager.arange(1, 10).reshape(3, 3);
//            nd = nd3.transpose();
//            nd = nd3.add(10);
//
//            /*
//            ND: (3, 3) cpu() int32
//            [[11, 14, 17],
//             [12, 15, 18],
//             [13, 16, 19],
//            ]
//            */
//
//
//        }

    }
}
