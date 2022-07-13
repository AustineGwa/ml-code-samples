import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;

public class Main {

    public static void main(String [] args){
        NDManager manager = NDManager.newBaseManager();
        NDArray x = manager.create(3f);
        NDArray y = manager.create(2f);
        x.add(y);
        System.out.println(x);

    }

}
