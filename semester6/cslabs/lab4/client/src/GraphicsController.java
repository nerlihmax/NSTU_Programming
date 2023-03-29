import network.NetworkEventListener;
import network.NetworkRepository;
import network.UDPNetworkRepository;
import objects.GraphicalObject;
import objects.Smiley;
import objects.Star;
import ui_components.ButtonsPanel;
import utils.EditorModes;
import utils.Vector;
import utils.network_events.*;

import javax.swing.*;
import java.awt.*;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;
import java.util.Scanner;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

public class GraphicsController extends JPanel implements NetworkEventListener, Runnable {
    private final ArrayList<GraphicalObject> objects = new ArrayList<>();
    private final ExecutorService executor = Executors.newSingleThreadExecutor();
    private final ScheduledExecutorService scheduledExecutor = Executors.newSingleThreadScheduledExecutor();
    private final Random random = new Random();
    private final NetworkRepository networkRepository;

    private EditorModes mode = EditorModes.ADD;

    private volatile boolean waitingForData = false;

    public GraphicsController(boolean isServer) {
        registerModesPanel();
        start();
        networkRepository = new UDPNetworkRepository(isServer, this);
        var thread = new Thread((Runnable) networkRepository);
        thread.setDaemon(true);
        thread.start();

        var mainThread = new Thread(this);
        mainThread.setDaemon(true);
        mainThread.start();
    }

    @Override
    protected void paintComponent(Graphics g) {
        super.paintComponent(g);
        for (GraphicalObject object : objects) {
            object.draw(g);
        }
    }

    private void registerModesPanel() {
        ButtonsPanel buttonsPanel = new ButtonsPanel();
        add(buttonsPanel, BorderLayout.WEST);

        buttonsPanel.onAddButtonClicked(e -> {
            mode = EditorModes.ADD;
            objects.forEach(GraphicalObject::hideOutline);
        });

        buttonsPanel.onRemoveButtonClicked(e -> {
            mode = EditorModes.REMOVE;
            objects.forEach(GraphicalObject::showOutline);
        });

        buttonsPanel.onStopButtonClicked(e -> {
            mode = EditorModes.STOP_RESUME;
            objects.forEach(GraphicalObject::showOutline);
        });

        buttonsPanel.onStopAllButtonClicked(e -> objects.forEach(GraphicalObject::stop));

        buttonsPanel.onResumeAllButtonClicked(e -> objects.forEach(GraphicalObject::resume));
    }

    private void start() {
        scheduledExecutor.scheduleAtFixedRate(() -> {
            executor.submit(() -> objects.forEach(obj -> {
                if (obj.isMoving()) {
                    obj.move(new Vector(getWidth(), getHeight()));
                }
            }));

            SwingUtilities.invokeLater(this::repaint);
        }, 0, 16, TimeUnit.MILLISECONDS);

        setBackground(Color.WHITE);
        addMouseListener(new MouseAdapter() {
            @Override
            public void mouseClicked(MouseEvent e) {
                int x = e.getX();
                int y = e.getY();

                GraphicalObject object = null;

                if (mode == EditorModes.REMOVE) {
                    objects.removeIf(obj -> obj.contains(x, y));
                    return;
                }

                if (mode == EditorModes.STOP_RESUME) {
                    objects.stream().filter(obj -> obj.contains(x, y)).findFirst().ifPresent(obj -> {
                        if (obj.isMoving()) {
                            obj.stop();
                        } else {
                            obj.resume();
                        }
                    });
                    return;
                }
                if (e.getButton() == MouseEvent.BUTTON1) {
                    object = new Smiley(x, y, 50, 50, Color.CYAN);
                } else if (e.getButton() == MouseEvent.BUTTON3) {
                    var size = random.nextInt(50, 150);
                    var vertices = random.nextInt(5, 9);
                    object = new Star(x, y, size, size, Color.RED, vertices);
                }

                if (object != null) {
                    objects.add(object);
                }
            }
        });

    }

    @Override
    public void onEvent(NetworkEvent event) {
        switch (event) {
            case ClearObjects ignored -> {
                objects.clear();
                repaint();
            }
            case ResponseObject responseObject -> {
                System.out.println("Received Object: " + responseObject.object());
                var object = switch (responseObject.type()) {
                    case "Star" -> new Star(100, 100, 100, 100, Color.RED);
                    case "Smiley" -> new Smiley(100, 100, 100, 100, Color.RED);
                    default -> throw new IllegalStateException("Unexpected value: " + responseObject.object());
                };
                object.readFromJson(responseObject.object());
                objects.add(object);
            }
            case ResponseObjectByIndex responseObjectByIndex -> {
                System.out.println("Received object with index: " + responseObjectByIndex.index());
                var object = switch (responseObjectByIndex.type()) {
                    case "Star" -> new Star(100, 100, 100, 100, Color.RED);
                    case "Smiley" -> new Smiley(100, 100, 100, 100, Color.RED);
                    default -> throw new IllegalStateException("Unexpected value: " + responseObjectByIndex.object());
                };
                object.readFromJson(responseObjectByIndex.object());
                objects.add(object);
            }

            case ResponseObjectListSize responseObjectListSize ->
                    System.out.println("Received objects list size: " + responseObjectListSize.size());
            case ResponseObjectList responseObjectList ->
                    System.out.println("Received objects list: " + Arrays.toString(responseObjectList.objects()));
            case RequestObjectList ignored -> {
                System.out.println("Requested object list");
                var objList = new GraphicalObject[objects.size()];
                networkRepository.sendObjectsList(objects.toArray(objList));
            }
            case RequestObjectListSize ignored -> {
                System.out.println("Requested object list size");
                networkRepository.sendObjectsListSize(objects.size());
            }
            case RequestObjectByIndex requestObjectByIndex -> {
                System.out.println("Requested object by index: " + requestObjectByIndex.index());
                networkRepository.sendObjectByIndex(requestObjectByIndex.index(), objects.get(requestObjectByIndex.index()));
            }
            default -> System.out.println("Unknown event: " + event);
        }
        System.out.println("============\n");
        waitingForData = false;
    }

    @Override
    public void run() {
        Scanner scanner = new Scanner(System.in);
        String input;
        boolean running = true;

        while (running) {
            if (waitingForData) {
                System.out.println("Waiting for response...");
                while (waitingForData) Thread.onSpinWait();
            }
            System.out.println("Please select an option:");
            System.out.println("1. Close connection");
            System.out.println("2. Clear objects");
            System.out.println("3. Request object by index");
            System.out.println("4. Request objects list");
            System.out.println("5. Request objects list size");
            System.out.println("6. Show local list");
            System.out.println("7. Send object by index");
            System.out.println("0. Exit");

            input = scanner.nextLine();
            waitingForData = true;
            switch (input) {
                case "1" -> {
                    networkRepository.closeConnection();
                    waitingForData = false;
                    System.out.println("============\n");
                }
                case "2" -> {
                    networkRepository.clearObjects();
                    objects.clear();
                    repaint();
                    waitingForData = false;
                    System.out.println("============\n");
                }
                case "3" -> {
                    System.out.print("Enter index: ");
                    input = scanner.nextLine();
                    networkRepository.requestObjectByIndex(Integer.parseInt(input));
                }
                case "4" -> networkRepository.requestObjectsList();
                case "5" -> networkRepository.requestObjectsListSize();
                case "6" -> {
                    System.out.println(objects.toString());
                    waitingForData = false;
                    System.out.println("============\n");
                }
                case "7" -> {
                    System.out.print("Enter index: ");
                    input = scanner.nextLine();
                    try {
                        var idx = Integer.parseInt(input);
                        var obj = objects.get(idx);
                        networkRepository.sendObjectByIndex(idx, obj);
                    } catch (Exception e) {
                        System.out.println("ERROR: " + e.getMessage());
                    }
                    waitingForData = false;
                    System.out.println("============\n");
                }
                case "0" -> running = false;
                default -> {
                    System.out.println("Invalid input, please try again.");
                    waitingForData = false;
                }
            }
        }
        scanner.close();
    }
}