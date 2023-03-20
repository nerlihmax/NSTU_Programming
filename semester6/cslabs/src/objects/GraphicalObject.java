package objects;

import utils.Vector;

import java.awt.*;
import java.io.*;

public abstract class GraphicalObject {
    protected int x, y;
    protected int width, height;
    protected Color color;

    private boolean isMoving = true;
    private boolean isShowOutline = false;

    public GraphicalObject(int x, int y, int width, int height, Color color) {
        this.x = x;
        this.y = y;
        this.width = width;
        this.height = height;
        this.color = color;
    }

    public void stop() {
        isMoving = false;
    }

    public void resume() {
        isMoving = true;
    }

    public boolean isMoving() {
        return isMoving;
    }

    public void showOutline() {
        isShowOutline = true;
    }

    public void hideOutline() {
        isShowOutline = false;
    }

    public boolean isShowOutline() {
        return isShowOutline;
    }

    public void draw(Graphics g) {
        if (isShowOutline) {
            g.setColor(Color.GREEN);
            g.drawRect(x - width / 2 - 1, y - height / 2 - 1, width + 2, height + 2);
        }
    }

    public boolean contains(int x, int y) {
        return (x >= this.x - width / 2 && x <= this.x + width / 2 && y >= this.y - height / 2 && y <= this.y + height / 2);
    }

    public void read(byte[] input) throws IOException {
        try (
                ByteArrayInputStream bis = new ByteArrayInputStream(input);
                ObjectInputStream dis = new ObjectInputStream(bis)) {
            x = dis.readInt();
            y = dis.readInt();
            width = dis.readInt();
            height = dis.readInt();
            int r = dis.readInt();
            int g = dis.readInt();
            int b = dis.readInt();
            color = new Color(r, g, b);
        }
    }

    public byte[] write() throws IOException {
        try (
                ByteArrayOutputStream bos = new ByteArrayOutputStream();
                ObjectOutputStream out = new ObjectOutputStream(bos)) {
            out.writeInt(x);
            out.writeInt(y);
            out.writeInt(width);
            out.writeInt(height);
            out.writeInt(color.getRed());
            out.writeInt(color.getGreen());
            out.writeInt(color.getBlue());
            return bos.toByteArray();
        }
    }

    public abstract void move(Vector movement);
}
