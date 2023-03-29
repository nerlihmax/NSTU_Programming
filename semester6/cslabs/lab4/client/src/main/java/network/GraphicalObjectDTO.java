package network;

import objects.GraphicalObject;
import objects.Smiley;
import objects.Star;

import java.awt.*;

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

    public static GraphicalObjectDTO fromGraphicalObject(GraphicalObject object) {
        return switch (object.getClass().getSimpleName()) {
            case "Star" ->
                    new StarDTO(object.getX(), object.getY(), object.getWidth(), object.getHeight(), object.getColor().getRed(), object.getColor().getGreen(), object.getColor().getBlue(), ((Star) object).getNumberOfVertices());
            case "Smiley" ->
                    new SmileyDTO(object.getX(), object.getY(), object.getWidth(), object.getHeight(), object.getColor().getRed(), object.getColor().getGreen(), object.getColor().getBlue(), ((Smiley) object).getVx(), ((Smiley) object).getVy());
            default -> throw new RuntimeException("Unknown type of graphical object");
        };
    }

    public GraphicalObject toGraphicalObject() {
        if (this instanceof StarDTO) {
            return new Star(x, y, width, height, new Color(r, g, b), ((StarDTO) this).numberOfVertices);
        } else if (this instanceof SmileyDTO) {
            return new Smiley(x, y, width, height, new Color(r, g, b), ((SmileyDTO) this).vx, ((SmileyDTO) this).vy);
        } else {
            throw new RuntimeException("Unknown type of graphical object");
        }
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
