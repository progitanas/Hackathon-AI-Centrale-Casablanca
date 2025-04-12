// Attend que le DOM soit entièrement chargé
document.addEventListener('DOMContentLoaded', () => {

    // Initialise la connexion Socket.IO
    const socket = io(); // Se connecte au serveur qui a servi la page

    // Éléments du DOM
    const turnEl = document.getElementById('turn');
    const aiLevelEl = document.getElementById('ai-level');
    const playerScoreEl = document.getElementById('player-score');
    const aiScoreEl = document.getElementById('ai-score');
    const zonesContainer = document.getElementById('zones-container');
    const actionsContainer = document.getElementById('actions-container');
    const confirmButton = document.getElementById('confirm-button');
    const startButton = document.getElementById('start-button');
    const levelSelect = document.getElementById('level');
    const messageEl = document.getElementById('message');
    const lastAiMoveEl = document.getElementById('last-ai-move');
    const aiHistoryListEl = document.getElementById('ai-history-list');
    const gameOverMessageEl = document.getElementById('game-over-message');
    const winnerMessageEl = document.getElementById('winner-message');

    // Variables d'état du frontend
    let selectedZone = null;
    let selectedAction = null;
    let currentZones = [];
    let currentActions = [];
    let isGameOver = false;

    // --- Fonctions de mise à jour de l'UI ---

    function updateUI(state) {
        console.log("Updating UI with state:", state); // Debug
        turnEl.textContent = state.turn;
        aiLevelEl.textContent = state.ai_level;
        playerScoreEl.textContent = state.player_score;
        aiScoreEl.textContent = state.ai_score;

        isGameOver = state.game_over;

        // Mettre à jour le dernier coup de l'IA
        if (state.last_ai_move) {
            lastAiMoveEl.textContent = `Dernier coup IA: ${state.last_ai_move[0]} - ${state.last_ai_move[1]}`;
        } else {
            lastAiMoveEl.textContent = 'Dernier coup IA: ---';
        }

        // Mettre à jour l'historique IA
        aiHistoryListEl.innerHTML = ''; // Vider la liste
        state.ai_history.slice().reverse().forEach((move, index) => { // Afficher les plus récents en premier
            const li = document.createElement('li');
            li.textContent = `T${move.turn}: ${move.zone} - ${move.action}`;
            if (index === 0) {
                li.classList.add('latest'); // Marquer le dernier coup
            }
            aiHistoryListEl.appendChild(li);
        });

        // Générer les boutons de zone si nécessaire (seulement une fois ou si ça change)
        if (currentZones.length === 0 && state.zones) {
            currentZones = state.zones;
            zonesContainer.innerHTML = '<h2>Choisissez une Zone :</h2>'; // Reset
            currentZones.forEach(zone => {
                const button = document.createElement('button');
                button.id = `zone-${zone}`;
                button.textContent = zone;
                button.dataset.zone = zone; // Stocker la valeur dans l'attribut data-*
                button.addEventListener('click', handleZoneClick);
                zonesContainer.appendChild(button);
            });
        }

         // Générer les boutons d'action si nécessaire
        if (currentActions.length === 0 && state.actions) {
            currentActions = state.actions;
            actionsContainer.innerHTML = '<h2>Choisissez une Action :</h2>'; // Reset
            currentActions.forEach(action => {
                const button = document.createElement('button');
                button.id = `action-${action}`;
                // Pourrait ajouter des icônes ici si souhaité
                button.textContent = action;
                button.dataset.action = action;
                button.addEventListener('click', handleActionClick);
                actionsContainer.appendChild(button);
            });
        }

        // Gérer l'état de fin de partie
        if (isGameOver) {
            messageEl.textContent = "Partie terminée !";
            confirmButton.disabled = true;
            disableActionButtons(); // Désactiver les boutons de jeu
            gameOverMessageEl.style.display = 'block'; // Afficher le message final
            if (state.winner === "Player") {
                winnerMessageEl.textContent = "Félicitations, vous avez gagné !";
            } else if (state.winner === "AI") {
                winnerMessageEl.textContent = "L'IA a gagné...";
            } else {
                winnerMessageEl.textContent = "Égalité !";
            }
        } else {
            gameOverMessageEl.style.display = 'none'; // Cacher le message final
            messageEl.textContent = "À vous de jouer ! Sélectionnez une zone et une action.";
            // Réactiver les boutons si nécessaire (après start_game)
            enableActionButtons();
            updateConfirmButtonState(); // Mettre à jour l'état du bouton confirmer
        }
        // Réinitialiser les sélections visuelles après chaque tour
        resetSelections();
    }

    function handleZoneClick(event) {
        if (isGameOver) return;
        selectedZone = event.target.dataset.zone;
        console.log("Zone selected:", selectedZone);
        // Mettre à jour l'UI pour montrer la sélection
        document.querySelectorAll('#zones-container button').forEach(btn => {
            btn.classList.remove('selected');
        });
        event.target.classList.add('selected');
        updateConfirmButtonState();
    }

    function handleActionClick(event) {
        if (isGameOver) return;
        selectedAction = event.target.dataset.action;
        console.log("Action selected:", selectedAction);
        // Mettre à jour l'UI
         document.querySelectorAll('#actions-container button').forEach(btn => {
            btn.classList.remove('selected');
        });
        event.target.classList.add('selected');
        updateConfirmButtonState();
    }

    function updateConfirmButtonState() {
        if (selectedZone && selectedAction && !isGameOver) {
            confirmButton.disabled = false;
        } else {
            confirmButton.disabled = true;
        }
    }

    function resetSelections() {
         selectedZone = null;
         selectedAction = null;
         document.querySelectorAll('#zones-container button, #actions-container button').forEach(btn => {
            btn.classList.remove('selected');
         });
         updateConfirmButtonState(); // Met à jour l'état du bouton (devrait le désactiver)
    }

     function disableActionButtons() {
        document.querySelectorAll('#zones-container button, #actions-container button').forEach(btn => {
           btn.disabled = true; // Désactiver pour le jeu
        });
        confirmButton.disabled = true;
     }

     function enableActionButtons() {
         document.querySelectorAll('#zones-container button, #actions-container button').forEach(btn => {
           btn.disabled = false;
        });
        // L'état de confirmButton dépendra des sélections
        updateConfirmButtonState();
     }


    // --- Gestionnaires d'événements SocketIO ---

    socket.on('connect', () => {
        console.log('Connected to server!');
        messageEl.textContent = 'Connecté au serveur. Choisissez un niveau et commencez.';
    });

    socket.on('disconnect', () => {
        console.log('Disconnected from server!');
        messageEl.textContent = 'Déconnecté du serveur ! Essayez de rafraîchir la page.';
        disableActionButtons(); // Désactiver si déconnecté
    });

    socket.on('update_state', (state) => {
        updateUI(state);
        // Après la mise à jour (qui peut avoir réinitialisé les sélections)
        // On s'assure que le bouton confirmer est dans le bon état
        updateConfirmButtonState();
    });

    socket.on('error_message', (data) => {
        // Pourrait être utilisé pour afficher des erreurs spécifiques du serveur
        console.error("Server error:", data.message);
        messageEl.textContent = `Erreur: ${data.message}`;
    });


    // --- Gestionnaires d'événements du DOM ---

    startButton.addEventListener('click', () => {
        const selectedLevel = levelSelect.value;
        console.log(`Starting new game with level: ${selectedLevel}`);
        // Réinitialiser l'UI avant de démarrer
        gameOverMessageEl.style.display = 'none';
        lastAiMoveEl.textContent = 'Dernier coup IA: ---';
        aiHistoryListEl.innerHTML = '';
        resetSelections();
        // Envoyer l'événement au serveur
        socket.emit('start_game', { level: selectedLevel });
        messageEl.textContent = `Nouvelle partie lancée (Niveau ${selectedLevel})...`;
        enableActionButtons(); // Réactiver les boutons pour la nouvelle partie
    });

    confirmButton.addEventListener('click', () => {
        if (selectedZone && selectedAction && !isGameOver) {
            console.log(`Confirming turn: Zone=${selectedZone}, Action=${selectedAction}`);
            // Envoyer l'action au serveur
            socket.emit('player_action', {
                zone: selectedZone,
                action: selectedAction
            });
            // Optionnel: désactiver les boutons en attendant la réponse
            messageEl.textContent = "Tour envoyé, attente de la réponse de l'IA...";
            disableActionButtons(); // Désactiver pendant le traitement
            // Les sélections seront réinitialisées par updateUI après la réponse
        } else {
            messageEl.textContent = "Veuillez sélectionner une zone ET une action.";
        }
    });

}); // Fin de DOMContentLoaded
