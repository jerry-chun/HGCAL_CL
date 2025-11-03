//Testing

//**************************************************
// \file HGCALTBDetConstruction.cc
// \brief: implementation of
//         HGCALTBDetConstruction class
// \author: Lorenzo Pezzotti (CERN EP-SFT-sim)
//          @lopezzot
// \start date: 10 January 2024
//**************************************************

// Includers from project files
//
#include "HGCALTBDetConstruction.hh"

#include "HGCALTBAHCALSD.hh"
#include "HGCALTBCEESD.hh"
#include "HGCALTBCHESD.hh"

// Includers from Geant4
//
#include "G4GDMLParser.hh"
#include "G4GeomTestVolume.hh"
#include "G4LogicalVolume.hh"
#include "G4LogicalVolumeStore.hh"
#include "G4PhysicalVolumeStore.hh"
#include "G4SDManager.hh"
#include "G4VisAttributes.hh"

// Includers from std
//
#include <string>
#include <cmath> 

std::vector<double> gLayerZ;

// Preprocessor macros
// #define CHECKOVERLAPS

// Constructors and de-constructor
//
HGCALTBDetConstruction::HGCALTBDetConstruction() : G4VUserDetectorConstruction() {}

HGCALTBDetConstruction::~HGCALTBDetConstruction() {}

// virtual Construct() method from base class
//
G4VPhysicalVolume* HGCALTBDetConstruction::Construct()
{
  G4GDMLParser Parser;
  Parser.Read("hgcalTB23May.gdml", false);
  auto worldPV = Parser.GetWorldVolume();

#if G4VERSION_NUMBER > 1100
#  ifdef CHECKOVERLAPS
  CheckOverlaps(worldPV);
#  endif
#endif

  DefineVisAttributes();

  return worldPV;
}

// virtual ConstructSDandField() method from base class
//
void HGCALTBDetConstruction::ConstructSDandField()
{
  // Sensitive detectors
  //
  auto CEESD = new HGCALTBCEESD("CEESD");
  G4SDManager::GetSDMpointer()->AddNewDetector(CEESD);
  auto CHESD = new HGCALTBCHESD("CHESD");
  G4SDManager::GetSDMpointer()->AddNewDetector(CHESD);
  auto AHCALSD = new HGCALTBAHCALSD("AHSD");
  G4SDManager::GetSDMpointer()->AddNewDetector(AHCALSD);

  // Assign to logical volume
  //

// Attach SDs only to true active sensors/tiles (by name + material).
// Skip passive wrappers like Absorber/Steel/AirGap/etc.
// ---- helpers as lambdas (valid inside a function) -------------------------
// helpers as lambdas (all inside the function)
auto Has = [](const G4String& s, const char* sub) -> bool {
  return s.find(sub) != std::string::npos;
};

auto IsSilicon = [&Has](const G4Material* m) -> bool {
  if (!m) return false;
  for (size_t i = 0; i < m->GetNumberOfElements(); ++i) {
    if (std::abs(m->GetElement(i)->GetZ() - 14.0) < 0.5) return true; // Si
  }
  const auto& n = m->GetName();
  return (n == "G4_Si") || Has(n, "Silicon");
};

auto IsScint = [&Has](const G4Material* m) -> bool {
  if (!m) return false;
  const auto& n = m->GetName();
  return Has(n, "SC_VINYLTOLUENE") || Has(n, "POLYSTYRENE") || Has(n, "Scint");
};

// ---- attach SDs only to true sensors/tiles --------------------------------
auto* store = G4LogicalVolumeStore::GetInstance();
G4cout << "\n[Audit] LV name | material | hasSD? | nDaughters\n";
for (auto* lv : *store) {
  auto* sd = lv->GetSensitiveDetector();
  const auto& n = lv->GetName();
  const auto* m = lv->GetMaterial();
  G4cout << "  " << n
         << " | " << (m? m->GetName() : "NULL")
         << " | " << (sd? "Y":"N")
         << " | " << lv->GetNoDaughters()
         << G4endl;
}

int nCEE = 0, nCHE = 0, nAH = 0, nSkip = 0;

for (auto* lv : *store) {
  const G4String& nm    = lv->GetName();
  const G4Material* mat = lv->GetMaterial();

  // We only attach to volumes that BOTH belong to the right family and contain "Sensitive"
  //const bool isEEcell = (Has(nm, "HGCalEE") && Has(nm, "Sensitive")) || IsSilicon(mat);
  //const bool isHEcell = (Has(nm, "HGCalHE") && Has(nm, "Sensitive")) || (IsSilicon(mat) || IsScint(mat));
  //const bool isAHtile = (Has(nm, "AHcal")   && Has(nm, "Sensitive")) || IsScint(mat);
    
  const bool isEEcell = Has(nm, "HGCalEE") && (Has(nm, "Sensitive"));
  const bool isHEcell = Has(nm, "HGCalHE") && (Has(nm, "Sensitive"));
  const bool isAHtile = Has(nm, "AHcal")   && (Has(nm, "Sensitive"));

  if (isEEcell) {
    lv->SetSensitiveDetector(CEESD);
    ++nCEE;
    G4cout << "[CEE] SD -> " << nm << " (mat=" << (mat?mat->GetName():"NULL") << ")\n";
    continue;
  }

  if (isHEcell) {
    lv->SetSensitiveDetector(CHESD);
    ++nCHE;
    G4cout << "[CHE] SD -> " << nm << " (mat=" << (mat?mat->GetName():"NULL") << ")\n";
    continue;
  }

  if (isAHtile) {
    lv->SetSensitiveDetector(AHCALSD);
    ++nAH;
    G4cout << "[AHCAL] SD -> " << nm << " (mat=" << (mat?mat->GetName():"NULL") << ")\n";
    continue;
  }

  ++nSkip;
}
    
if (gLayerZ.empty()) {
  auto* pvStore = G4PhysicalVolumeStore::GetInstance();
  for (auto* pv : *pvStore) {
    const auto& n = pv->GetName();
    bool isLayer =
        (n.find("HGCalEELayerF")          != G4String::npos) ||
        (n.find("HGCalEELayerB")          != G4String::npos) ||
        (n.find("HGCalHESiliconLayer")    != G4String::npos) ||
        (n.find("HGCalHEScintillatorLayer")!= G4String::npos);
    if (!isLayer) continue;

    double z_cm = pv->GetTranslation().z() / CLHEP::cm;
    gLayerZ.push_back(z_cm);
  }
  std::sort(gLayerZ.begin(), gLayerZ.end());
  gLayerZ.erase(std::unique(gLayerZ.begin(), gLayerZ.end(),
                            [](double a, double b){ return std::fabs(a-b) < 0.05; }),
                gLayerZ.end());

  G4cout << "[LayerMap] collected " << gLayerZ.size()
         << " layer z-positions\n";
}

G4cout << "[SD] attached: CEE=" << nCEE
       << " CHE=" << nCHE
       << " AHCAL=" << nAH
       << " (skipped " << nSkip << " others)\n";


if (auto* hcTable = G4SDManager::GetSDMpointer()->GetHCtable()) {
  G4cout << "[HCtable] entries=" << hcTable->entries() << G4endl;
  for (int i = 0; i < hcTable->entries(); ++i)
    G4cout << "  [" << i << "] " << hcTable->GetHCname(i) << G4endl;
}

//auto LVStore = G4LogicalVolumeStore::GetInstance();
//for (auto volume : *LVStore) {
//    const G4String& name = volume->GetName();
//
  //  if (name.find("HGCalEECellSensitive") != std::string::npos) {
      //  G4cout << "[CEE] Assigned to: " << name << G4endl;
    //    volume->SetSensitiveDetector(CEESD);
    }
 //   if (name.find("HGCalEESensitive") != std::string::npos) {
  //      G4cout << "[CEE] Assigned to: " << name << G4endl;
   //     volume->SetSensitiveDetector(CEESD);
 //   }

//    if (name.find("HGCalEE") != std::string::npos) {
//	G4cout <<"[CEE] Assigned"  << name << G4endl;
//	volume->SetSensitiveDetector(CEESD);
  //  }
  //  if (name.find("HGCalHECellSensitive") != std::string::npos || name.find("HGCalHESiliconSensitive") != std::string::npos || name.find("HGCalHESiliconCellSensitive") != std::string::npos) {
  //      G4cout << "[CHE] Assigned to: " << name << G4endl;
  //      volume->SetSensitiveDetector(CHESD);
 //  }
 
//  if (name.find("HGCalHESensitive") != std::string::npos){
  //      G4cout << "[CHE] Assigned to: " << name << G4endl;
  //      volume->SetSensitiveDetector(CHESD);
  //  }
 //   if (name.find("HGCalHESilicon") != std::string::npos){
 //       G4cout << "[CHE] Assigned to: " << name << G4endl;
 //       volume->SetSensitiveDetector(CHESD);
 //   }

  //  if (name.find("HGCalHE") != std::string::npos) {
  //      G4cout <<"[CHE] Assigned"  << name << G4endl;
  //      volume->SetSensitiveDetector(CHESD);
  //  }

//    if (name.find("AHcalTileSensitive") != std::string::npos) {
//        G4cout << "[AHCAL] Assigned to: " << name << G4endl;
//        volume->SetSensitiveDetector(AHCALSD);
//    }
//} 



  // No fields involved
//}

// DefineVisAttributes() private method
//
void HGCALTBDetConstruction::DefineVisAttributes()
{
  // Lambda function to check if a string is included in another
  auto isSubstring = [](const std::string& mainString, const std::string& searchString) {
    return mainString.find(searchString) != std::string::npos;
  };

  // VisAttributes definitions
  auto SiWaferVisAttr = new G4VisAttributes();
  SiWaferVisAttr->SetForceSolid(true);
  SiWaferVisAttr->SetColor(G4Color::Green());
  SiWaferVisAttr->SetDaughtersInvisible(true);
  auto TotalInvisibleVisAttr = new G4VisAttributes();
  TotalInvisibleVisAttr->SetVisibility(false);
  TotalInvisibleVisAttr->SetDaughtersInvisible(true);
  auto TileVisAttr = new G4VisAttributes();
  TileVisAttr->SetForceSolid(false);
  TileVisAttr->SetColor(G4Color::Red());
  TileVisAttr->SetDaughtersInvisible(true);

  // Assign vis attributes
  //
  auto LVStore = G4LogicalVolumeStore::GetInstance();
  for (auto volume : *LVStore) {
    // beam line
    if (volume->GetName() == "HGCalBeam") volume->SetVisAttributes(TotalInvisibleVisAttr);
    if (volume->GetName() == "HGCalBeamDown") volume->SetVisAttributes(TotalInvisibleVisAttr);
    if (volume->GetName() == "HGCalBeamS5") volume->SetVisAttributes(TotalInvisibleVisAttr);
    if (volume->GetName() == "HGCalBeamS6") volume->SetVisAttributes(TotalInvisibleVisAttr);
    if (volume->GetName() == "CMSE") volume->SetVisAttributes(G4VisAttributes::GetInvisible());
    if (volume->GetName() == "HGCal") volume->SetVisAttributes(G4VisAttributes::GetInvisible());
    // cee
    if (volume->GetName() == "HGCalEE") volume->SetVisAttributes(G4VisAttributes::GetInvisible());
    if (isSubstring(volume->GetName(), "HGCalEEBlock"))
      volume->SetVisAttributes(G4VisAttributes::GetInvisible());
    if (isSubstring(volume->GetName(), "HGCalEEgap"))
      volume->SetVisAttributes(G4VisAttributes::GetInvisible());
    if (isSubstring(volume->GetName(), "HGCalEEAlcase"))
      volume->SetVisAttributes(G4VisAttributes::GetInvisible());
    if (isSubstring(volume->GetName(), "HGCalEEAbsorber"))
      volume->SetVisAttributes(G4VisAttributes::GetInvisible());
    if (isSubstring(volume->GetName(), "HGCalEECuPCB"))
      volume->SetVisAttributes(G4VisAttributes::GetInvisible());
    if (isSubstring(volume->GetName(), "HGCalEEPCB"))
      volume->SetVisAttributes(G4VisAttributes::GetInvisible());
    if (isSubstring(volume->GetName(), "HGCalEECuKapton"))
      volume->SetVisAttributes(G4VisAttributes::GetInvisible());
    if (isSubstring(volume->GetName(), "HGCalEESensitive"))
      volume->SetVisAttributes(SiWaferVisAttr);
    // che
    if (volume->GetName() == "HGCalHE") volume->SetVisAttributes(G4VisAttributes::GetInvisible());
    if (isSubstring(volume->GetName(), "HGCalHEBlock"))
      volume->SetVisAttributes(G4VisAttributes::GetInvisible());
    if (isSubstring(volume->GetName(), "HGCalHEgap"))
      volume->SetVisAttributes(G4VisAttributes::GetInvisible());
    if (isSubstring(volume->GetName(), "HGCalHEAbsorber"))
      volume->SetVisAttributes(G4VisAttributes::GetInvisible());
    if (isSubstring(volume->GetName(), "HGCalHECuPCB"))
      volume->SetVisAttributes(G4VisAttributes::GetInvisible());
    if (isSubstring(volume->GetName(), "HGCalHEPCB"))
      volume->SetVisAttributes(G4VisAttributes::GetInvisible());
    if (isSubstring(volume->GetName(), "HGCalHECuKapton"))
      volume->SetVisAttributes(G4VisAttributes::GetInvisible());
    if (isSubstring(volume->GetName(), "HGCalHESiliconSensitive"))
      volume->SetVisAttributes(SiWaferVisAttr);
    // ahcal
    if (volume->GetName() == "HGCalAH") volume->SetVisAttributes(G4VisAttributes::GetInvisible());
    if (isSubstring(volume->GetName(), "AHcalTileSensitive")) volume->SetVisAttributes(TileVisAttr);
  }
}

// CheckOverlaps() private method
//
#if G4VERSION_NUMBER > 1100
void HGCALTBDetConstruction::CheckOverlaps(G4VPhysicalVolume* PhysVol)
{
  G4cout << "-->CheckingOverlaps for volumes in " << PhysVol->GetName() << G4endl;
  // volume, tolerance, npoints, verbosity
  G4GeomTestVolume* testVolume = new G4GeomTestVolume(PhysVol, 10.0, 100000, true);
  testVolume->TestOverlapInTree();
}
#endif

//**************************************************
