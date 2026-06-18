#ifndef HGCALTBCHESD_h
#define HGCALTBCHESD_h 1

#include "G4VSensitiveDetector.hh"
#include "HGCALTBCHEHit.hh"

#include <unordered_map>
#include <vector>
#include <utility>

// Forward declarations
class G4Step;
class G4HCofThisEvent;

// Key to identify a unique CHE cell:
// layer index + integerised center coordinates (in cm).
struct CHECellKey {
  int layer;
  int ix;  // rounded x center [cm]
  int iy;  // rounded y center [cm]

  bool operator==(const CHECellKey& other) const {
    return (layer == other.layer) && (ix == other.ix) && (iy == other.iy);
  }
};

// Hash for CHECellKey so we can use it in std::unordered_map
struct CHECellKeyHash {
  std::size_t operator()(const CHECellKey& k) const noexcept {
    std::size_t h1 = std::hash<int>{}(k.layer);
    std::size_t h2 = std::hash<int>{}(k.ix);
    std::size_t h3 = std::hash<int>{}(k.iy);
    std::size_t h  = h1 ^ (h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2));
    h ^= (h3 + 0x9e3779b9 + (h << 6) + (h >> 2));
    return h;
  }
};

// Accumulator for one CHE cell over an event
struct CHECellAccum {
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

class HGCALTBCHESD : public G4VSensitiveDetector {
public:
    HGCALTBCHESD(const G4String& name);
    virtual ~HGCALTBCHESD() = default;

    virtual void Initialize(G4HCofThisEvent* HCE) override;
    virtual G4bool ProcessHits(G4Step* aStep, G4TouchableHistory* ROhist) override;
    virtual void EndOfEvent(G4HCofThisEvent* HCE) override;

    static const G4String fCHEHitsCollectionName;

private:
    HGCALTBCHEHitsCollection* fHitsCollection = nullptr;

    // Per-event map of (layer, ix, iy) -> accumulated cell info
    std::unordered_map<CHECellKey, CHECellAccum, CHECellKeyHash> fCellMap;

    // (still here if you ever want wafer IDs, but not used in merging)
    G4int FindWaferID(G4int cpno) const;

    const std::unordered_map<G4int, G4int> CHEWaferMap = {
        {0, 0}, {10002, 1}, {100101, 2}, {101, 3},
        {10101, 4}, {110101, 5}, {2, 6},
        {33, 7}, {34, 8}, {35, 9}, {36, 10}, {37, 11}, {38, 12}, {39, 13}
    };
};

#endif
