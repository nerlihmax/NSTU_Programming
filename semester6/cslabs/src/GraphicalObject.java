import java.awt.*;
import java.io.*;

public abstract class GraphicalObject {
    protected int x, y; // координаты центра
    protected int width, height; // размеры охватывающего прямоугольника
    protected Color color; // цвет

    public GraphicalObject(int x, int y, int width, int height, Color color) {
        this.x = x;
        this.y = y;
        this.width = width;
        this.height = height;
        this.color = color;
    }

    public abstract void draw(Graphics g, int canvasWidth, int canvasHeight); // рисование объекта

    public boolean contains(int x, int y) {
        return (x >= this.x - width / 2 && x <= this.x + width / 2 && y >= this.y - height / 2 && y <= this.y + height / 2);
    } // проверка на принадлежность точки объекту

    public void read(InputStream input) throws IOException {
        var dis = new DataInputStream(input);
        x = dis.readInt();
        y = dis.readInt();
        width = dis.readInt();
        height = dis.readInt();
        int r = dis.readInt();
        int g = dis.readInt();
        int b = dis.readInt();
        color = new Color(r, g, b);
    } // чтение из потока

    public void write(OutputStream output) throws IOException {
        var dos = new DataOutputStream(output);
        dos.writeInt(x);
        dos.writeInt(y);
        dos.writeInt(width);
        dos.writeInt(height);
        dos.writeInt(color.getRed());
        dos.writeInt(color.getGreen());
        dos.writeInt(color.getBlue());
    }

    public abstract void move(Vector movement);
}
