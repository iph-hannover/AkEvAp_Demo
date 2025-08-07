import subprocess, os
from types import SimpleNamespace
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QApplication, QWidget
from bibs.guis import gui_userInputs
from pose_classification import PoseTypeLMM as TYPE

class userwidget(QWidget, gui_userInputs.Ui_Form):
    finish = pyqtSignal()
    def __init__(self, liftNrs, mainLift):
        super(userwidget, self).__init__()
        self.setupUi(self)
        self.setGui(liftNrs, mainLift)
        self.btn_info.clicked.connect(self.showInfo)
        self.lineE_wiederholung.textChanged.connect(self.updateWiederholung)
        self.lineE_lastGewicht.textChanged.connect(self.updateLastGewicht)
        self.cBox_gender.currentTextChanged.connect(self.updateLastGewicht)
        self.rbtn_anfForm0.toggled.connect(self.updateBody)
        self.rbtn_anfForm1.toggled.connect(self.updateBody)
        self.rbtn_anfForm2.toggled.connect(self.updateBody)
        self.rbtn_anfForm3.toggled.connect(self.updateBody)
        self.rbtn_endForm0.toggled.connect(self.updateBody)
        self.rbtn_endForm1.toggled.connect(self.updateBody)
        self.rbtn_endForm2.toggled.connect(self.updateBody)
        self.rbtn_endForm3.toggled.connect(self.updateBody)

        self.rbtn_aufnahmeGut.toggled.connect(self.updateErgebnis)
        self.rbtn_aufnahmeSchlecht.toggled.connect(self.updateErgebnis)
        self.rbtn_aufnahmeSehrSchlecht.toggled.connect(self.updateErgebnis)

        self.rbtn_greiferGut.toggled.connect(self.updateErgebnis)
        self.rbtn_greiferSchlecht.toggled.connect(self.updateErgebnis)
        self.rbtn_greiferSehrSchlecht.toggled.connect(self.updateErgebnis)

        self.rbtn_handNie.toggled.connect(self.updateErgebnis)
        self.rbtn_handMittel.toggled.connect(self.updateErgebnis)
        self.rbtn_handViel.toggled.connect(self.updateErgebnis)

        self.rbtn_witterungGut.toggled.connect(self.updateErgebnis)
        self.rbtn_witterungSchlecht.toggled.connect(self.updateErgebnis)

        self.rbtn_orgaGut.toggled.connect(self.updateErgebnis)
        self.rbtn_orgaSchlecht.toggled.connect(self.updateErgebnis)
        self.rbtn_orgaSehrSchlecht.toggled.connect(self.updateErgebnis)

        self.rbtn_haltenGut.toggled.connect(self.updateErgebnis)
        self.rbtn_haltenSchlecht.toggled.connect(self.updateErgebnis)
        self.rbtn_haltenSehrSchlecht.toggled.connect(self.updateErgebnis)

        self.rbtn_kleidungGut.toggled.connect(self.updateErgebnis)
        self.rbtn_kleidungSchlecht.toggled.connect(self.updateErgebnis)

        self.rbtn_raumGut.toggled.connect(self.updateErgebnis)
        self.rbtn_raumSchlecht.toggled.connect(self.updateErgebnis)
        self.rbtn_raumSehrSchlecht.toggled.connect(self.updateErgebnis)

    def setGui(self, liftNrs, mainLift):
        self.label_40.setText('*')
        pixmapForm0 = QPixmap(os.path.join('bibs', 'guis', 'form0.png'))
        pixmapForm1 = QPixmap(os.path.join('bibs', 'guis', 'form1.png'))
        pixmapForm2 = QPixmap(os.path.join('bibs', 'guis', 'form2.png'))
        pixmapForm3 = QPixmap(os.path.join('bibs', 'guis', 'form3.png'))
        self.lab_anfForm0.setPixmap(pixmapForm0)
        self.lab_anfForm1.setPixmap(pixmapForm1)
        self.lab_anfForm2.setPixmap(pixmapForm2)
        self.lab_anfForm3.setPixmap(pixmapForm3)
        self.lab_endForm0.setPixmap(pixmapForm0)
        self.lab_endForm1.setPixmap(pixmapForm1)
        self.lab_endForm2.setPixmap(pixmapForm2)
        self.lab_endForm3.setPixmap(pixmapForm3)
        if mainLift[0] == 0:
            self.rbtn_anfForm0.setChecked(True)
        elif mainLift[0] == 1:
            self.rbtn_anfForm1.setChecked(True)
        elif mainLift[0] == 2:
            self.rbtn_anfForm2.setChecked(True)
        elif mainLift[0] == 3:
            self.rbtn_anfForm3.setChecked(True)

        if mainLift[1] == 0:
            self.rbtn_endForm0.setChecked(True)
        elif mainLift[1] == 1:
            self.rbtn_endForm1.setChecked(True)
        elif mainLift[1] == 2:
            self.rbtn_endForm2.setChecked(True)
        elif mainLift[1] == 3:
            self.rbtn_endForm3.setChecked(True)
        self.updateBody()
        self.lineE_wiederholung.setText(str(liftNrs))
        self.updateWiederholung()
        self.updateLastGewicht()
        self.updateErgebnis()

    def updateErgebnis(self):
        gewichtungen = self.getRbtnValues()
        bodyGewichtung = int(self.lineE_gewichtBody.text())
        zeitGewichtung = int(self.lineE_gewichtZeit.text())
        lastGewichtung = int(self.lineE_gewicht_4.text())

        self.lineE_auswertungLast.setText(str(lastGewichtung))
        self.lineE_auswertungAufnahme.setText(str(gewichtungen.aufnahmeGewichtung))
        self.lineE_auswertungBody.setText(str(bodyGewichtung))
        bedingungen = (gewichtungen.handGewichtung + gewichtungen.greifGewichtung + gewichtungen.witterungGewichtung + gewichtungen.haltenGewichtung +
            gewichtungen.kleidungGewichtung + gewichtungen.raumGewichtung)
        self.lineE_auswertungBedingunggen.setText(str(bedingungen))
        self.lineE_auswertungOrga.setText(str(gewichtungen.orgaGewichtung))
        self.lineE_auswertungZeit.setText(str(zeitGewichtung))
        self.lineE_auswertungSumme.setText(str(bedingungen + lastGewichtung + bodyGewichtung + gewichtungen.aufnahmeGewichtung
                                               + gewichtungen.orgaGewichtung))
        self.lineE_auswertungErgebnis.setText(str(int(self.lineE_auswertungSumme.text()) * zeitGewichtung))
        risikoColor, risikoValue = self.getColor(int(self.lineE_auswertungErgebnis.text()))
        self.btn_risiko.setStyleSheet("background-color:" + risikoColor)
        self.lineE_risiko.setText(str(risikoValue))

    def getColor(self, ergebnisValue):
        if ergebnisValue >= 100:
            risikoColor = 'red'
            risikoValue = 4
        elif ergebnisValue >= 50 and ergebnisValue < 100:
            risikoColor = 'yellow'
            risikoValue = 3
        elif ergebnisValue >= 20 and ergebnisValue < 50:
            risikoColor = 'limegreen'
            risikoValue = 2
        else:
            risikoColor = 'green'
            risikoValue = 1
        return risikoColor, risikoValue

    def getRbtnValues(self):
        gewichtungen = SimpleNamespace(
            handGewichtung = self.getHandGewichtung(),
            greifGewichtung = self.getGreifGewichtung(),
            witterungGewichtung = self.getWitterungGewichtung(),
            haltenGewichtung = self.getHaltenGewichtung(),
            kleidungGewichtung = self.getKleidungGewichtung(),
            raumGewichtung = self.getRaumGewichtung(),
            orgaGewichtung = self.getOrgaGewichtung(),
            aufnahmeGewichtung = self.getAufnahmeGewichtung(),
        )
        return gewichtungen

    def getAufnahmeGewichtung(self):
        if self.rbtn_aufnahmeGut.isChecked():
            return 0
        elif self.rbtn_aufnahmeSchlecht.isChecked():
            return 1
        else:
            return 2

    def getOrgaGewichtung(self):
        if self.rbtn_orgaGut.isChecked():
            return 0
        elif self.rbtn_orgaSchlecht.isChecked():
            return 2
        else:
            return 4

    def getRaumGewichtung(self):
        if self.rbtn_raumGut.isChecked():
            return 0
        elif self.rbtn_raumSchlecht.isChecked():
            return 1
        else:
            return 2

    def getKleidungGewichtung(self):
        if self.rbtn_kleidungGut.isChecked():
            return 0
        else:
            return 1

    def getHaltenGewichtung(self):
        if self.rbtn_haltenGut.isChecked():
            return 0
        elif self.rbtn_haltenSchlecht.isChecked():
            return 2
        else:
            return 5

    def getWitterungGewichtung(self):
        if self.rbtn_witterungGut.isChecked():
            return 0
        else:
            return 1

    def getGreifGewichtung(self):
        if self.rbtn_greiferGut.isChecked():
            return 0
        elif self.rbtn_greiferSchlecht.isChecked():
            return 1
        else:
            return 2

    def getHandGewichtung(self):
        if self.rbtn_handNie.isChecked():
            return 0
        elif self.rbtn_handMittel.isChecked():
            return 1
        else:
            return 2

    def updateLastGewicht(self):
        if self.lineE_lastGewicht.text() == '':
            lastGewicht = 0
        else:
            lastGewicht = float(self.lineE_lastGewicht.text())
        if self.cBox_gender.currentText() == 'Frau':
            if lastGewicht < 3.0:
                self.lineE_gewicht_4.setText('0')
            elif lastGewicht >= 3.0 and lastGewicht < 5.0:
                self.lineE_gewicht_4.setText('6')
            elif lastGewicht >= 5.0 and lastGewicht < 10.0:
                self.lineE_gewicht_4.setText('9')
            elif lastGewicht >= 10.0 and lastGewicht < 15.0:
                self.lineE_gewicht_4.setText('12')
            elif lastGewicht >= 15.0 and lastGewicht < 20.0:
                self.lineE_gewicht_4.setText('25')
            elif lastGewicht >= 20.0 and lastGewicht < 25.0:
                self.lineE_gewicht_4.setText('75')
            elif lastGewicht >= 25.0 and lastGewicht < 30.0:
                self.lineE_gewicht_4.setText('85')
            else:
                self.lineE_gewicht_4.setText('100')
        else:
            if lastGewicht < 3.0:
                self.lineE_gewicht_4.setText('0')
            elif lastGewicht >= 3.0 and lastGewicht < 5.0:
                self.lineE_gewicht_4.setText('4')
            elif lastGewicht >= 5.0 and lastGewicht < 10.0:
                self.lineE_gewicht_4.setText('6')
            elif lastGewicht >= 10.0 and lastGewicht < 15.0:
                self.lineE_gewicht_4.setText('8')
            elif lastGewicht >= 15.0 and lastGewicht < 20.0:
                self.lineE_gewicht_4.setText('11')
            elif lastGewicht >= 20.0 and lastGewicht < 25.0:
                self.lineE_gewicht_4.setText('15')
            elif lastGewicht >= 25.0 and lastGewicht < 30.0:
                self.lineE_gewicht_4.setText('25')
            elif lastGewicht >= 30.0 and lastGewicht < 35.0:
                self.lineE_gewicht_4.setText('35')
            elif lastGewicht >= 35.0 and lastGewicht < 40.0:
                self.lineE_gewicht_4.setText('75')
            else:
                self.lineE_gewicht_4.setText('100')
        self.updateErgebnis()

    def updateBody(self):
        if self.rbtn_anfForm0.isChecked():
            bodyAnfang = 0
        elif self.rbtn_anfForm1.isChecked():
            bodyAnfang = 1
        elif self.rbtn_anfForm2.isChecked():
            bodyAnfang = 2
        elif self.rbtn_anfForm3.isChecked():
            bodyAnfang = 3
        else:
            bodyAnfang = 4

        if self.rbtn_endForm0.isChecked():
            bodyEnde = 0
        elif self.rbtn_endForm1.isChecked():
            bodyEnde = 1
        elif self.rbtn_endForm2.isChecked():
            bodyEnde = 2
        elif self.rbtn_endForm3.isChecked():
            bodyEnde = 3
        else:
            bodyEnde = 4

        bodyAnfang = TYPE(bodyAnfang)
        bodyEnde = TYPE(bodyEnde)

        if bodyAnfang == TYPE.NORMAL and bodyEnde == TYPE.NORMAL:
            self.lineE_gewichtBody.setText('0')
        elif (bodyAnfang == TYPE.NORMAL and bodyEnde == TYPE.SLIGHT_OFFSET) or (bodyAnfang == TYPE.SLIGHT_OFFSET and bodyEnde == TYPE.NORMAL):
            self.lineE_gewichtBody.setText('3')
        elif bodyAnfang == TYPE.SLIGHT_OFFSET and bodyEnde == TYPE.SLIGHT_OFFSET:
            self.lineE_gewichtBody.setText('5')
        elif (bodyAnfang == TYPE.NORMAL and bodyEnde == TYPE.STRONG_OFFSET) or (bodyAnfang == TYPE.STRONG_OFFSET and bodyEnde == TYPE.NORMAL):
            self.lineE_gewichtBody.setText('7')
        elif (bodyAnfang == TYPE.NORMAL and bodyEnde == TYPE.KNEEING) or (bodyAnfang == TYPE.KNEEING and bodyEnde == TYPE.NORMAL):
            self.lineE_gewichtBody.setText('9')
        elif (bodyAnfang == TYPE.SLIGHT_OFFSET and bodyEnde == TYPE.STRONG_OFFSET) or (bodyAnfang == TYPE.STRONG_OFFSET and bodyEnde == TYPE.SLIGHT_OFFSET):
            self.lineE_gewichtBody.setText('10')
        elif (bodyAnfang == TYPE.SLIGHT_OFFSET and bodyEnde == TYPE.KNEEING) or (bodyAnfang == TYPE.KNEEING and bodyEnde == TYPE.SLIGHT_OFFSET):
            self.lineE_gewichtBody.setText('13')
        elif bodyAnfang == TYPE.STRONG_OFFSET and bodyEnde == TYPE.STRONG_OFFSET:
            self.lineE_gewichtBody.setText('15')
        elif (bodyAnfang == TYPE.STRONG_OFFSET and bodyEnde == TYPE.KNEEING) or (bodyAnfang == TYPE.KNEEING and bodyEnde == TYPE.STRONG_OFFSET):
            self.lineE_gewichtBody.setText('18')
        elif bodyAnfang == TYPE.KNEEING and bodyEnde == TYPE.KNEEING:
            self.lineE_gewichtBody.setText('20')
        else:
            self.lineE_gewichtBody.setText('100')
        self.updateErgebnis()

    def updateWiederholung(self):
        if self.lineE_wiederholung.text() == '':
            widerholungsValue = 0
        else:
            widerholungsValue = int(self.lineE_wiederholung.text())
        if widerholungsValue < 7.5:
            self.lineE_gewichtZeit.setText('1')
        elif widerholungsValue >= 7.5 and widerholungsValue < 35:
            self.lineE_gewichtZeit.setText('1.5')
        elif widerholungsValue >= 35 and widerholungsValue < 63:
            self.lineE_gewichtZeit.setText('2')
        elif widerholungsValue >= 63 and widerholungsValue < 125:
            self.lineE_gewichtZeit.setText('2.5')
        elif widerholungsValue >= 125 and widerholungsValue < 185:
            self.lineE_gewichtZeit.setText('3')
        elif widerholungsValue >= 185 and widerholungsValue < 255:
            self.lineE_gewichtZeit.setText('3.5')
        elif widerholungsValue >= 255 and widerholungsValue < 400:
            self.lineE_gewichtZeit.setText('4')
        elif widerholungsValue >= 400 and widerholungsValue < 625:
            self.lineE_gewichtZeit.setText('5')
        elif widerholungsValue >= 625 and widerholungsValue < 875:
            self.lineE_gewichtZeit.setText('6')
        elif widerholungsValue >= 875 and widerholungsValue < 1250:
            self.lineE_gewichtZeit.setText('7')
        elif widerholungsValue >= 1250 and widerholungsValue < 1750:
            self.lineE_gewichtZeit.setText('8')
        elif widerholungsValue >= 1750 and widerholungsValue < 2250:
            self.lineE_gewichtZeit.setText('9')
        else:
            self.lineE_gewichtZeit.setText('10')
        self.updateErgebnis()

    def showInfo(self):
        subprocess.Popen([os.path.join('bibs', 'guis', 'handBewegungBeispiel.png')], shell=True)
