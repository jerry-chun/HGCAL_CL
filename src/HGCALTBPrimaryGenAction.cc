// HGCALTBPrimaryGenAction.cc
// --------------------------------------------------
#include "HGCALTBPrimaryGenAction.hh"
#include "G4ParticleGun.hh"

// Geant4
#include "G4Event.hh"
#include "G4ParticleTable.hh"
#include "G4SystemOfUnits.hh"
#include "G4ThreeVector.hh"
#include "Randomize.hh"
#include "G4PhysicalConstants.hh"

// STL parsing
#include <sstream>

HGCALTBPrimaryGenAction::HGCALTBPrimaryGenAction()
  : G4VUserPrimaryGeneratorAction()
  , fParticleGun(new G4ParticleGun(1))
  , fMessenger(this, "/HGCAL/gun/", "Primary generator control")
{
    // Default ranges (min/max differ by a small amount)
    fNumParticlesMin   = 1;
    fNumParticlesMax   = 2;
    fEnergyMin         = 10.0 * GeV;
    fEnergyMax         = 11.0 * GeV;
    fThetaMin          = 0.0 * rad;
    fThetaMax          = 1.0 * deg;
    fZMin              = -900.0 * cm;
    fZMax              = -899.0 * cm;
    fParticleTypesStr  = "pi+";

    // UI commands
    fMessenger.DeclareProperty("numParticlesMin", fNumParticlesMin,
                                "Minimum # of particles per event");
    fMessenger.DeclareProperty("numParticlesMax", fNumParticlesMax,
                                "Maximum # of particles per event");
    fMessenger.DeclarePropertyWithUnit("energyMin", "GeV", fEnergyMin,
                                        "Minimum particle energy");
    fMessenger.DeclarePropertyWithUnit("energyMax", "GeV", fEnergyMax,
                                        "Maximum particle energy");
    fMessenger.DeclarePropertyWithUnit("thetaMin", "deg", fThetaMin,
                                        "Minimum polar angle");
    fMessenger.DeclarePropertyWithUnit("thetaMax", "deg", fThetaMax,
                                        "Maximum polar angle");
    fMessenger.DeclarePropertyWithUnit("zMin", "cm", fZMin,
                                        "Minimum z position");
    fMessenger.DeclarePropertyWithUnit("zMax", "cm", fZMax,
                                        "Maximum z position");
    fMessenger.DeclareProperty("particleTypes", fParticleTypesStr,
                                "Space-separated list of particle names (e.g. \"mu- mu+\")");
}

HGCALTBPrimaryGenAction::~HGCALTBPrimaryGenAction()
{
    delete fParticleGun;
}

std::vector<G4String> HGCALTBPrimaryGenAction::ParseParticleTypes() const
{
    std::vector<G4String> types;
    std::istringstream iss(fParticleTypesStr);
    G4String token;
    while (iss >> token) {
        types.push_back(token);
    }
    return types;
}

void HGCALTBPrimaryGenAction::GeneratePrimaries(G4Event* event)
{
    auto pTable = G4ParticleTable::GetParticleTable();
    
    fPrimaryEnergiesMeV.clear();
    fPrimaryPDGIDs.clear();
    
    // random N in [min, max]
    G4int nTracks = G4RandFlat::shootInt(fNumParticlesMin, fNumParticlesMax + 1);

    auto types = ParseParticleTypes();
    if (types.empty()) {
        G4cerr << "[PrimaryGen] ERROR: no particle types specified!\n";
        return;
    }

    for (G4int i = 0; i < nTracks; ++i) {
        // choose particle type randomly
        G4int idx = static_cast<G4int>(
            G4RandFlat::shootInt(
                static_cast<long>(0),
                static_cast<long>(types.size())
            )
        );
        G4String name = types[idx];
        auto pDef = pTable->FindParticle(name);
        if (!pDef) {
            G4cerr << "[PrimaryGen] ERROR: unknown particle '" << name << "'\n";
            continue;
        }

        // energy, angle, phi, z
        G4double E     = G4RandFlat::shoot(fEnergyMin, fEnergyMax);
// Sample theta uniformly in solid angle
	G4double cosThetaMin = std::cos(fThetaMin);
	G4double cosThetaMax = std::cos(fThetaMax);
	G4double cosTheta = G4RandFlat::shoot(cosThetaMax, cosThetaMin);
	G4double theta = std::acos(cosTheta);
    //    G4double theta = G4RandFlat::shoot(fThetaMin, fThetaMax);
        G4double phi   = G4RandFlat::shoot(0., twopi);
        G4ThreeVector dir(std::sin(theta)*std::cos(phi),
                          std::sin(theta)*std::sin(phi),
                          std::cos(theta));
        G4double zPos  = G4RandFlat::shoot(fZMin, fZMax);
// *** DEBUG PRINT ***
    G4cout
      << "[PrimaryGen] dir = ("
      << dir.x() << ", "
      << dir.y() << ", "
      << dir.z() << ")   "
      << "θ=" << theta/deg << "°  "
      << "φ=" << phi/deg << "°"
      << G4endl;
    // *** END DEBUG 
        
        fPrimaryEnergiesMeV.push_back(E / CLHEP::MeV);
        fPrimaryPDGIDs.push_back(pDef->GetPDGEncoding());

        // configure and shoot
        fParticleGun->SetParticleDefinition(pDef);
        fParticleGun->SetParticleEnergy(E);
        fParticleGun->SetParticleMomentumDirection(dir);
        fParticleGun->SetParticlePosition({0., 0., zPos});
        fParticleGun->GeneratePrimaryVertex(event);
    }
}
