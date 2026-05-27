/**
 * @file firmware.ino
 * @brief Clean protocol: UPPERCASE to turn ON, lowercase to turn OFF.
 * - Pin 12: Green LED (G/g)
 * - Pin 11: Red LED + Buzzer (R/r)
 * - Pin 10: Blue LED (B/b)
 */

const int GREEN_LED_PIN = 12;   
const int RED_LED_PIN   = 11;   
const int BLUE_LED_PIN  = 10;   

void setup() {
  pinMode(GREEN_LED_PIN, OUTPUT);
  pinMode(RED_LED_PIN, OUTPUT);
  pinMode(BLUE_LED_PIN, OUTPUT);

  digitalWrite(GREEN_LED_PIN, LOW);
  digitalWrite(RED_LED_PIN, LOW);
  digitalWrite(BLUE_LED_PIN, LOW); 

  Serial.begin(9600);
}

void loop() {
  if (Serial.available() > 0) {
    char command = Serial.read(); 

    switch (command) {
      // --- GREEN LED (Tracking) ---
      case 'G': digitalWrite(GREEN_LED_PIN, HIGH); break;
      case 'g': digitalWrite(GREEN_LED_PIN, LOW);  break;

      // --- RED LED + BUZZER (Fall) ---
      case 'R': digitalWrite(RED_LED_PIN, HIGH);   break;
      case 'r': digitalWrite(RED_LED_PIN, LOW);    break;

      // --- BLUE LED (System / Extra) ---
      case 'B': digitalWrite(BLUE_LED_PIN, HIGH);  break;
      case 'b': digitalWrite(BLUE_LED_PIN, LOW);   break;

      default:  break; 
    }
  }
}