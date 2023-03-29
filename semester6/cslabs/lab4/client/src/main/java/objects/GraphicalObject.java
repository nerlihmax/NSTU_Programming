package objects;

import utils.Vector;

import java.awt.*;

public abstract class GraphicalObject {
    protected int x, y;
    protected int width, height;
    protected Color color;

    public int getX() {
        return x;
    }

    public int getY() {
        return y;
    }

    public int getWidth() {
        return width;
    }

    public int getHeight() {
        return height;
    }

    public Color getColor() {
        return color;
    }

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

    public abstract void move(Vector movement);
}
