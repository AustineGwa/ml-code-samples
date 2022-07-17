package ai;

import ai.djl.ndarray.NDManager;
import tech.tablesaw.api.BooleanColumn;
import tech.tablesaw.api.DoubleColumn;
import tech.tablesaw.api.StringColumn;
import tech.tablesaw.api.Table;
import tech.tablesaw.columns.Column;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.List;

public class PlayGround {

    //Creating and handling missing data in datset using tablesaw
    static void testing() throws IOException {
        File file = new File("../data/");
        file.mkdir();

        String dataFile = "../data/house_tiny.csv";

// Create file
        File f = new File(dataFile);
        f.createNewFile();

// Write to file
        try (FileWriter writer = new FileWriter(dataFile)) {
            writer.write("NumRooms,Alley,Price\n"); // Column names
            writer.write("NA,Pave,127500\n");  // Each row represents a data example
            writer.write("2,NA,106000\n");
            writer.write("4,NA,178100\n");
            writer.write("NA,NA,140000\n");
        }

        Table data = Table.read().file("../data/house_tiny.csv");
        Table inputs = data.create(data.columns());
        inputs.removeColumns("Price");
        Table outputs = data.select("Price");

        Column col = inputs.column("NumRooms");
        col.set(col.isMissing(), (int) inputs.nCol("NumRooms").mean());

        StringColumn col2= (StringColumn) inputs.column("Alley");
        List<BooleanColumn> dummies = col2.getDummies();
        inputs.removeColumns(col2);
        inputs.addColumns(DoubleColumn.create("Alley_Pave", dummies.get(0).asDoubleArray()),
                DoubleColumn.create("Alley_nan", dummies.get(1).asDoubleArray())
        );
        System.out.println(inputs);

        //conversion to NDArray format
        NDManager nd = NDManager.newBaseManager();
        NDArray x = (NDArray) nd.create(inputs.as().doubleMatrix());
        NDArray y = (NDArray) nd.create(outputs.as().intMatrix());
        System.out.println(x);
        System.out.println(y);
    }


    public static void main(String[] args) {
        try{
            testing();

        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }
}
