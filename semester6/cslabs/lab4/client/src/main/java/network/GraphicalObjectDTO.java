package network;

public class GraphicalObjectDTO {
    protected int x, y, width, height, r, g, b;

    public GraphicalObjectDTO(int x, int y, int width, int height, int r, int g, int b) {
        this.x = x;
        this.y = y;
        this.width = width;
        this.height = height;
        this.r = r;
        this.g = g;
        this.b = b;
    }
}

class StarDTO extends GraphicalObjectDTO {
    protected int numberOfVertices;

    public StarDTO(int x, int y, int width, int height, int r, int g, int b, int numberOfVertices) {
        super(x, y, width, height, r, g, b);
        this.numberOfVertices = numberOfVertices;
    }
}

class SmileyDTO extends GraphicalObjectDTO {
    protected int vx, vy;

    public SmileyDTO(int x, int y, int width, int height, int r, int g, int b, int vx, int vy) {
        super(x, y, width, height, r, g, b);
        this.vx = vx;
        this.vy = vy;
    }
}
