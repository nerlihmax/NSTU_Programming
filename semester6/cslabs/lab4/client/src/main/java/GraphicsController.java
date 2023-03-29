import network.NetworkEventListener;
import network.NetworkRepository;
import network.RESTNetworkRepository;
import objects.GraphicalObject;
import objects.Smiley;
import objects.Star;
import ui_components.ButtonsPanel;
import utils.EditorModes;
import utils.Vector;
import utils.network_events.NetworkEvent;
import utils.network_events.ResponseObject;
import utils.network_events.ResponseObjectList;
import utils.network_events.ResponseObjectListNames;

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

    public GraphicsController() {
        registerModesPanel();
        start();
        networkRepository = new RESTNetworkRepository(this);

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

        buttonsPanel.onClearButtonClicked(e -> objects.clear());
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
            case ResponseObject responseObject -> {
                objects.add(responseObject.object());
            }
            case ResponseObjectList responseObjectList -> {
                objects.addAll(Arrays.asList(responseObjectList.objects()));
            }
            case ResponseObjectListNames responseObjectListNames -> {
                System.out.println("Objects list names:");
                for (GraphicalObject object : responseObjectListNames.objects()) {
                    System.out.println(object.toString());
                }
            }
            default -> System.out.println("Unknown event: " + event);
        }

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
            System.out.println("============\n");
            System.out.println("Please select an option:");
            System.out.println("1. Request object by index");
            System.out.println("2. Request objects list names");
            System.out.println("3. Synchronize objects list");
            System.out.println("4. Remove object by index");
            System.out.println("5. Add object by index");
            System.out.println("0. Exit");

            input = scanner.nextLine();
            waitingForData = true;
            switch (input) {
                case "1" -> {
                    System.out.print("Enter index: ");
                    input = scanner.nextLine();
                    networkRepository.requestObjectByIndex(Integer.parseInt(input));
                }
                case "2" -> networkRepository.requestObjectsListNames();
                case "3" -> networkRepository.requestObjectsList();

                case "4" -> {
                    System.out.print("Enter index: ");
                    input = scanner.nextLine();
                    objects.remove(Integer.parseInt(input));
                    networkRepository.removeObjectByIndex(Integer.parseInt(input));
                    waitingForData = false;
                }

                case "5" -> {
                    System.out.print("Enter index: ");
                    input = scanner.nextLine();
                    try {
                        var obj = objects.get(Integer.parseInt(input));
                        networkRepository.sendObject(obj);
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