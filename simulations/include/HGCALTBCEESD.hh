//**************************************************
// \file HGCALTBCEESD.hh
// \brief: definition of HGCALTBCEESD class
//**************************************************

#ifndef HGCALTBCEESD_h
#  define HGCALTBCEESD_h 1

// Includers from Geant4
//
#  include "G4VSensitiveDetector.hh"

// Includers from project files
//
#  include "HGCALTBCEEHit.hh"

// Includers from std
//
#  include <unordered_map>
#  include <vector>

// Forward declaration from Geant4
//
class G4Step;
class G4HCofThisEvent;

// Key to identify a unique cell in the EE:
// layer index + cell copy number.
// Key to identify a unique cell in the EE:
// layer index + integerised center coordinates (in cm).
struct CEECellKey {
  int layer;
  int ix;  // rounded x center [cm]
  int iy;  // rounded y center [cm]

  bool operator==(const CEECellKey& other) const {
    return (layer == other.layer) && (ix == other.ix) && (iy == other.iy);
  }
};

// Hash for CEECellKey so we can use it in std::unordered_map
struct CEECellKeyHash {
  std::size_t operator()(const CEECellKey& k) const noexcept {
    std::size_t h1 = std::hash<int>{}(k.layer);
    std::size_t h2 = std::hash<int>{}(k.ix);
    std::size_t h3 = std::hash<int>{}(k.iy);
    // standard hash combine
    std::size_t h = h1 ^ (h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2));
    h ^= (h3 + 0x9e3779b9 + (h << 6) + (h >> 2));
    return h;
  }
};

// Accumulator for one cell over an event
struct CEECellAccum {
  double edep = 0.0;  // total energy in this cell

  // true geometric center of this cell in *cm* (global coordinates)
  double cx = 0.0;
  double cy = 0.0;
  double cz = 0.0;
  bool   hasCenter = false;

  int firstTrackID = -1;  // track ID of the first step seen in this cell

  // (showerID, edep_from_this_shower) so we can pick the dominant shower
  std::vector<std::pair<int, double>> showerContribs;
};


class HGCALTBCEESD : public G4VSensitiveDetector
{
  public:
    HGCALTBCEESD(const G4String& name);
    virtual ~HGCALTBCEESD();

    // virtual methods from base class
    //
    virtual void Initialize(G4HCofThisEvent* hitCollection) override;
    virtual G4bool ProcessHits(G4Step* aStep, G4TouchableHistory* history) override;
    virtual void EndOfEvent(G4HCofThisEvent* hitCollection) override;

    // This sensitive detector creates 1 hit collection
    //
    static const G4String fCEEHitsCollectionName;

  private:
    HGCALTBCEEHitsCollection* fHitsCollection = nullptr;

    // Map (layer, cellCopy) -> accumulated energy/position over the event
    std::unordered_map<CEECellKey, CEECellAccum, CEECellKeyHash> fCellMap;
    double fStepEdepSum = 0.0;
};

#endif  // HGCALTBCEESD_h 1

//**************************************************
