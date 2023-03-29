package network;

import utils.network_events.NetworkEvent;

public interface NetworkEventListener {
    void onEvent(NetworkEvent event);
}
