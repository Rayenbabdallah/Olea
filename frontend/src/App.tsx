/**
 * OLEA AI — Cognitive Insurance Decision Engine
 * Frontend connected to FastAPI backend (Phase II)
 */

import React, { useState } from 'react';
import {
  Brain,
  Shield,
  Activity,
  TrendingUp,
  Users,
  BarChart3,
  Zap,
  Target,
  Layers,
  RefreshCw,
  Info,
  ArrowUpRight,
  ArrowDownRight,
  Sparkles,
  Settings2,
  Lock,
  ChevronDown,
  ChevronUp,
  FlaskConical,
  MessageSquareQuote,
} from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';

// ═══════════════════════════════════════════════════════════════════════
//  Types — matching backend schemas exactly
// ═══════════════════════════════════════════════════════════════════════

const BUNDLE_NAMES: Record<number, string> = {
  0: 'Auto Comprehensive',
  1: 'Auto Liability Basic',
  2: 'Basic Health',
  3: 'Family Comprehensive',
  4: 'Health Dental Vision',
  5: 'Home Premium',
  6: 'Home Standard',
  7: 'Premium Health Life',
  8: 'Renter Basic',
  9: 'Renter Premium',
};

interface InfluenceItem {
  feature: string;
  label: string;
  impact: number;
  direction: string;
}

interface UncertaintyInfo {
  entropy: number;
  level: string;
  explanation: string;
}

interface NearestBundleItem {
  bundle: number;
  bundle_name: string;
  similarity_level: string;
  similarity_context: string;
}

interface NarrativeStep {
  step: number;
  agent: string;
  title: string;
  summary: string;
}

interface AgentVote {
  agent: string;
  top_bundle: number;
  top_bundle_name: string;
  confidence: number;
  probabilities: number[];
}

interface AgentConsensus {
  top_bundle: number;
  top_bundle_name: string;
  confidence: number;
  agreement: number;
  total_agents: number;
  probabilities: number[];
}

interface BusinessInsight {
  customer_segment: string;
  retention_risk: string;
  upsell_potential: string;
}

interface ExplainResponse {
  predicted_bundle: number;
  confidence: number;
  probabilities: number[];
  uncertainty: UncertaintyInfo;
  top_influences: InfluenceItem[];
  nearest_bundles: NearestBundleItem[];
  narrative: { decision_timeline: NarrativeStep[] };
  agent_votes: AgentVote[];
  agent_consensus: AgentConsensus;
  upsell_suggestions: string[];
  business_insight: BusinessInsight;
}

interface SimulateResponse {
  base: { predicted_bundle: number; confidence: number; probabilities: number[] };
  modified: { predicted_bundle: number; confidence: number; probabilities: number[] };
  delta: {
    bundle_changed: boolean;
    top_probability_shift: { class: number; from: number; to: number; delta: number };
  };
  upsell_suggestions: string[];
}

interface CustomerData {
  Policy_Start_Year: number;
  Policy_Start_Month: string;
  Policy_Start_Week: number;
  Policy_Start_Day: number;
  Grace_Period_Extensions: number;
  Previous_Policy_Duration_Months: number;
  Adult_Dependents: number;
  Child_Dependents: number;
  Infant_Dependents: number;
  Region_Code: string;
  Existing_Policyholder: number;
  Previous_Claims_Filed: number;
  Years_Without_Claims: number;
  Policy_Amendments_Count: number;
  Underwriting_Processing_Days: number;
  Vehicles_on_Policy: number;
  Custom_Riders_Requested: number;
  Broker_Agency_Type: string;
  Deductible_Tier: string;
  Acquisition_Channel: string;
  Payment_Schedule: string;
  Employment_Status: string;
  Estimated_Annual_Income: number;
  Days_Since_Quote: number;
}

// ═══════════════════════════════════════════════════════════════════════
//  Constants — categorical dropdown options from training data
// ═══════════════════════════════════════════════════════════════════════

const DEDUCTIBLE_OPTIONS = ['Tier_4_Zero_Ded', 'Tier_3_Low_Ded', 'Tier_2_Mid_Ded', 'Tier_1_High_Ded'];
const CHANNEL_OPTIONS = ['Direct_Website', 'Local_Broker', 'Aggregator_Site', 'Corporate_Partner', 'Affiliate_Group'];
const PAYMENT_OPTIONS = ['Monthly_EFT', 'Quarterly_Invoice', 'Annual_Upfront'];
const EMPLOYMENT_OPTIONS = ['Employed_FullTime', 'Self_Employed', 'Contractor', 'Unemployed'];
const BROKER_OPTIONS = ['National_Corporate', 'Urban_Boutique'];
const MONTH_OPTIONS = ['January','February','March','April','May','June','July','August','September','October','November','December'];
const REGION_OPTIONS = ['ZAF','NGA','KEN','GHA','EGY','MAR','TZA','ETH','UGA','RWA','SEN','CIV','CMR','MOZ','TUN','DZA','BWA','NAM','MUS','AGO','USA','GBR','FRA','DEU','IND','BRA','CHN'];

// ═══════════════════════════════════════════════════════════════════════
//  Default customer profile
// ═══════════════════════════════════════════════════════════════════════

const DEFAULT_CUSTOMER: CustomerData = {
  Policy_Start_Year: 2016,
  Policy_Start_Month: 'February',
  Policy_Start_Week: 8,
  Policy_Start_Day: 15,
  Grace_Period_Extensions: 1,
  Previous_Policy_Duration_Months: 12,
  Adult_Dependents: 1,
  Child_Dependents: 2,
  Infant_Dependents: 0,
  Region_Code: 'ZAF',
  Existing_Policyholder: 1,
  Previous_Claims_Filed: 1,
  Years_Without_Claims: 3,
  Policy_Amendments_Count: 2,
  Underwriting_Processing_Days: 14,
  Vehicles_on_Policy: 1,
  Custom_Riders_Requested: 1,
  Broker_Agency_Type: 'Urban_Boutique',
  Deductible_Tier: 'Tier_3_Low_Ded',
  Acquisition_Channel: 'Direct_Website',
  Payment_Schedule: 'Monthly_EFT',
  Employment_Status: 'Self_Employed',
  Estimated_Annual_Income: 55000,
  Days_Since_Quote: 5,
};

// ═══════════════════════════════════════════════════════════════════════
//  Sub-components
// ═══════════════════════════════════════════════════════════════════════

function AgentNode({ name, vote, active, delay }: { name: string; vote: string; active: boolean; delay: number }) {
  return (
    <motion.div
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ delay }}
      className={`relative flex items-center p-4 rounded-xl border transition-all duration-500 ${
        active
          ? 'bg-olea-orange/10 border-olea-orange/30 shadow-[0_0_15px_rgba(242,159,65,0.1)]'
          : 'bg-white/50 border-olea-dark/5 grayscale opacity-50'
      }`}
    >
      <div className={`w-10 h-10 rounded-full flex items-center justify-center mr-4 ${active ? 'bg-olea-orange text-white' : 'bg-slate-200 text-slate-400'}`}>
        <Brain className="w-5 h-5" />
      </div>
      <div className="flex-1 min-w-0">
        <div className="text-[10px] text-slate-400 font-mono uppercase tracking-tighter">{name}</div>
        <div className="text-sm font-semibold text-olea-dark truncate">{vote}</div>
      </div>
      {active && (
        <motion.div
          layoutId="active-indicator"
          className="absolute -right-2 top-1/2 -translate-y-1/2 w-4 h-4 bg-olea-orange rounded-full blur-sm"
        />
      )}
    </motion.div>
  );
}

function SliderInput({
  label,
  value,
  min,
  max,
  step,
  format,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  format?: (v: number) => string;
  onChange: (v: number) => void;
}) {
  return (
    <div className="space-y-2">
      <div className="flex justify-between text-[10px] font-black uppercase tracking-widest">
        <label className="text-slate-500">{label}</label>
        <span className="text-olea-terracotta">{format ? format(value) : value}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full h-1 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-olea-orange"
      />
    </div>
  );
}

function SelectInput({
  label,
  value,
  options,
  format,
  onChange,
}: {
  label: string;
  value: string;
  options: string[];
  format?: (v: string) => string;
  onChange: (v: string) => void;
}) {
  return (
    <div className="space-y-2">
      <label className="text-[10px] font-black text-slate-500 uppercase tracking-widest block">{label}</label>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="w-full text-xs bg-olea-bg border border-olea-dark/10 rounded-lg px-3 py-2.5 text-olea-dark font-medium focus:outline-none focus:border-olea-orange transition-colors appearance-none cursor-pointer"
      >
        {options.map((o) => (
          <option key={o} value={o}>
            {format ? format(o) : o.replace(/_/g, ' ')}
          </option>
        ))}
      </select>
    </div>
  );
}

function ToggleInput({
  label,
  value,
  onChange,
}: {
  label: string;
  value: number;
  onChange: (v: number) => void;
}) {
  return (
    <div className="flex items-center justify-between">
      <label className="text-[10px] font-black text-slate-500 uppercase tracking-widest">{label}</label>
      <button
        onClick={() => onChange(value === 1 ? 0 : 1)}
        className={`relative w-10 h-5 rounded-full transition-colors ${value === 1 ? 'bg-olea-orange' : 'bg-slate-300'}`}
      >
        <div className={`absolute top-0.5 w-4 h-4 rounded-full bg-white shadow transition-transform ${value === 1 ? 'translate-x-5' : 'translate-x-0.5'}`} />
      </button>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════
//  Main App
// ═══════════════════════════════════════════════════════════════════════

export default function App() {
  const [loading, setLoading] = useState(false);
  const [step, setStep] = useState(0);
  const [result, setResult] = useState<ExplainResponse | null>(null);
  const [simResult, setSimResult] = useState<SimulateResponse | null>(null);
  const [customer, setCustomer] = useState<CustomerData>({ ...DEFAULT_CUSTOMER });
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const updateField = <K extends keyof CustomerData>(key: K, val: CustomerData[K]) => {
    setCustomer((prev) => ({ ...prev, [key]: val }));
  };

  // ── Run full analysis via /explain ─────────────────────────────────
  const handleAnalyze = async () => {
    setLoading(true);
    setStep(0);
    setError(null);
    setSimResult(null);

    try {
      const res = await fetch('/api/explain', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(customer),
      });

      if (!res.ok) {
        const detail = await res.text();
        throw new Error(`API error ${res.status}: ${detail}`);
      }

      const data: ExplainResponse = await res.json();
      setResult(data);

      // Animate agent timeline
      for (let i = 1; i <= 5; i++) {
        await new Promise((r) => setTimeout(r, 600));
        setStep(i);
      }
    } catch (err: any) {
      console.error(err);
      setError(err.message || 'Failed to connect to backend');
    } finally {
      setLoading(false);
    }
  };

  // ── Simulate field change via /simulate ────────────────────────────
  const handleSimulate = async (field: keyof CustomerData, newVal: CustomerData[keyof CustomerData]) => {
    if (!result) return;

    try {
      const res = await fetch('/api/simulate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          base: customer,
          changes: { [field]: newVal },
        }),
      });

      if (res.ok) {
        const data: SimulateResponse = await res.json();
        setSimResult(data);
      }
    } catch (err) {
      console.error('Simulation error:', err);
    }
  };

  // ── Helpers ────────────────────────────────────────────────────────
  const bundleName = (idx: number) => BUNDLE_NAMES[idx] ?? `Bundle ${idx}`;
  const pct = (v: number) => `${(v * 100).toFixed(1)}%`;

  return (
    <div className="min-h-screen relative overflow-hidden bg-olea-bg">
      {/* Background Pattern */}
      <div className="absolute inset-0 african-pattern pointer-events-none" />

      {/* Top Brand Bar */}
      <div className="bg-olea-dark py-2 px-6 flex justify-between items-center text-white/80 text-[10px] font-mono tracking-widest uppercase relative z-20">
        <div className="flex gap-4">
          <span>Pan-African Insurance Network</span>
          <span className="text-olea-orange">● 26 Countries</span>
        </div>
        <div className="flex gap-6">
          <div className="flex gap-4">
            <span className="cursor-pointer hover:text-white">Our Network</span>
            <span className="cursor-pointer hover:text-white">About</span>
            <span className="cursor-pointer hover:text-white">Contact</span>
          </div>
          <div className="flex gap-2">
            <span className="bg-olea-yellow text-olea-dark px-2 py-0.5 rounded font-bold cursor-pointer">Customer Area</span>
            <span className="border border-white/20 px-2 py-0.5 rounded cursor-pointer">English ▾</span>
          </div>
        </div>
      </div>

      <div className="p-6 lg:p-10 max-w-[1600px] mx-auto relative z-10">
        {/* Header */}
        <header className="flex flex-col md:flex-row md:items-center justify-between mb-10 gap-6">
          <div className="flex items-center gap-6">
            <div className="relative w-16 h-16">
              <div className="absolute inset-0 border-[8px] border-olea-orange rounded-full" style={{ clipPath: 'polygon(0 0, 100% 0, 100% 50%, 0 50%)' }} />
              <div className="absolute inset-0 border-[8px] border-olea-terracotta rounded-full" style={{ clipPath: 'polygon(0 50%, 100% 50%, 100% 100%, 0 100%)' }} />
              <div className="absolute inset-0 flex items-center justify-center">
                <Shield className="w-6 h-6 text-olea-dark" />
              </div>
            </div>
            <div>
              <h1 className="text-4xl font-serif font-black tracking-tighter text-olea-dark flex items-baseline gap-1">
                OLEA <span className="text-olea-orange font-light italic">Intelligence</span>
              </h1>
              <p className="text-olea-terracotta/80 font-mono text-xs uppercase tracking-[0.2em] mt-1">
                Your Panafrican Insurance Decision Engine
              </p>
            </div>
          </div>

          <div className="flex items-center gap-4">
            <div className="hidden md:flex flex-col items-end mr-4">
              <span className="text-[10px] text-slate-400 uppercase font-bold tracking-wider">Multi-Agent System</span>
              <span className="text-sm font-bold text-olea-dark">4 Specialist Agents • LightGBM</span>
            </div>
            <button className="glass px-4 py-2 rounded-xl text-sm font-medium text-olea-dark flex items-center gap-2 hover:bg-white transition-all border-olea-dark/10">
              <Settings2 className="w-4 h-4" />
              Settings
            </button>
            <button
              onClick={handleAnalyze}
              disabled={loading}
              className="olea-gradient hover:opacity-90 text-white px-8 py-3 rounded-xl text-sm font-bold flex items-center gap-2 shadow-xl shadow-olea-orange/20 transition-all active:scale-95 disabled:opacity-50"
            >
              {loading ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Zap className="w-4 h-4" />}
              RUN COGNITIVE ANALYSIS
            </button>
          </div>
        </header>

        {/* Error Banner */}
        {error && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            className="mb-6 p-4 bg-rose-50 border border-rose-200 rounded-xl text-rose-700 text-sm font-medium flex items-center gap-3"
          >
            <Info className="w-5 h-5 flex-shrink-0" />
            {error}
            <button onClick={() => setError(null)} className="ml-auto text-rose-400 hover:text-rose-600">✕</button>
          </motion.div>
        )}

        <main className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* ═══════════════════════════════════════════════════════════
              LEFT COLUMN: Input Form
          ═══════════════════════════════════════════════════════════ */}
          <div className="lg:col-span-3 space-y-6">
            <section className="glass rounded-2xl p-6 neo-border relative overflow-hidden bg-white/90">
              <div className="scanline" />
              <h3 className="text-xs font-black text-olea-dark uppercase tracking-[0.2em] mb-6 flex items-center gap-3 border-b border-olea-dark/5 pb-4">
                <Users className="w-4 h-4 text-olea-orange" />
                Customer Profile
              </h3>

              <div className="space-y-6">
                {/* Income */}
                <SliderInput
                  label="Annual Income"
                  value={customer.Estimated_Annual_Income}
                  min={0}
                  max={500000}
                  step={5000}
                  format={(v) => `$${v.toLocaleString()}`}
                  onChange={(v) => updateField('Estimated_Annual_Income', v)}
                />

                {/* Dependents */}
                <div className="grid grid-cols-3 gap-3">
                  <SliderInput label="Adults" value={customer.Adult_Dependents} min={0} max={10} step={1} onChange={(v) => updateField('Adult_Dependents', v)} />
                  <SliderInput label="Children" value={customer.Child_Dependents} min={0} max={10} step={1} onChange={(v) => updateField('Child_Dependents', v)} />
                  <SliderInput label="Infants" value={customer.Infant_Dependents} min={0} max={5} step={1} onChange={(v) => updateField('Infant_Dependents', v)} />
                </div>

                {/* Employment */}
                <SelectInput label="Employment" value={customer.Employment_Status} options={EMPLOYMENT_OPTIONS} onChange={(v) => updateField('Employment_Status', v)} />

                {/* Vehicles */}
                <SliderInput label="Vehicles on Policy" value={customer.Vehicles_on_Policy} min={0} max={8} step={1} onChange={(v) => updateField('Vehicles_on_Policy', v)} />

                {/* Deductible */}
                <SelectInput
                  label="Deductible Tier"
                  value={customer.Deductible_Tier}
                  options={DEDUCTIBLE_OPTIONS}
                  format={(v) => v.replace('Tier_', 'T').replace('_Ded', '').replace(/_/g, ' ')}
                  onChange={(v) => updateField('Deductible_Tier', v)}
                />

                {/* Region */}
                <SelectInput label="Region" value={customer.Region_Code} options={REGION_OPTIONS} onChange={(v) => updateField('Region_Code', v)} />

                {/* Toggles */}
                <ToggleInput label="Existing Policyholder" value={customer.Existing_Policyholder} onChange={(v) => updateField('Existing_Policyholder', v)} />

                {/* Claims */}
                <div className="grid grid-cols-2 gap-3">
                  <SliderInput label="Claims Filed" value={customer.Previous_Claims_Filed} min={0} max={15} step={1} onChange={(v) => updateField('Previous_Claims_Filed', v)} />
                  <SliderInput label="Claim-Free Yrs" value={customer.Years_Without_Claims} min={0} max={30} step={1} onChange={(v) => updateField('Years_Without_Claims', v)} />
                </div>
              </div>

              {/* Advanced toggle */}
              <button
                onClick={() => setShowAdvanced(!showAdvanced)}
                className="w-full mt-6 py-2 text-[10px] font-black text-olea-dark/50 uppercase tracking-widest flex items-center justify-center gap-2 border-t border-olea-dark/5 pt-4 hover:text-olea-orange transition-colors"
              >
                {showAdvanced ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
                {showAdvanced ? 'Hide' : 'Show'} Advanced Fields
              </button>

              <AnimatePresence>
                {showAdvanced && (
                  <motion.div
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ height: 'auto', opacity: 1 }}
                    exit={{ height: 0, opacity: 0 }}
                    className="overflow-hidden"
                  >
                    <div className="space-y-5 pt-4">
                      <SelectInput label="Channel" value={customer.Acquisition_Channel} options={CHANNEL_OPTIONS} onChange={(v) => updateField('Acquisition_Channel', v)} />
                      <SelectInput label="Payment" value={customer.Payment_Schedule} options={PAYMENT_OPTIONS} onChange={(v) => updateField('Payment_Schedule', v)} />
                      <SelectInput label="Broker Type" value={customer.Broker_Agency_Type} options={BROKER_OPTIONS} onChange={(v) => updateField('Broker_Agency_Type', v)} />
                      <SelectInput label="Policy Month" value={customer.Policy_Start_Month} options={MONTH_OPTIONS} onChange={(v) => updateField('Policy_Start_Month', v)} />
                      <SliderInput label="Policy Year" value={customer.Policy_Start_Year} min={2015} max={2017} step={1} onChange={(v) => updateField('Policy_Start_Year', v)} />
                      <SliderInput label="Policy Duration (mo)" value={customer.Previous_Policy_Duration_Months} min={0} max={50} step={1} onChange={(v) => updateField('Previous_Policy_Duration_Months', v)} />
                      <SliderInput label="Amendments" value={customer.Policy_Amendments_Count} min={0} max={10} step={1} onChange={(v) => updateField('Policy_Amendments_Count', v)} />
                      <SliderInput label="Grace Extensions" value={customer.Grace_Period_Extensions} min={0} max={10} step={1} onChange={(v) => updateField('Grace_Period_Extensions', v)} />
                      <SliderInput label="Underwriting Days" value={customer.Underwriting_Processing_Days} min={0} max={100} step={1} onChange={(v) => updateField('Underwriting_Processing_Days', v)} />
                      <SliderInput label="Days Since Quote" value={customer.Days_Since_Quote} min={0} max={200} step={1} onChange={(v) => updateField('Days_Since_Quote', v)} />
                      <SliderInput label="Custom Riders" value={customer.Custom_Riders_Requested} min={0} max={5} step={1} onChange={(v) => updateField('Custom_Riders_Requested', v)} />
                      <SliderInput label="Start Week" value={customer.Policy_Start_Week} min={1} max={53} step={1} onChange={(v) => updateField('Policy_Start_Week', v)} />
                      <SliderInput label="Start Day" value={customer.Policy_Start_Day} min={1} max={31} step={1} onChange={(v) => updateField('Policy_Start_Day', v)} />
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </section>

            {/* Upsell / Simulation Panel */}
            <section className="bg-olea-dark rounded-2xl p-6 shadow-xl relative overflow-hidden">
              <div className="absolute top-0 right-0 w-32 h-32 bg-olea-orange/10 rounded-full -mr-16 -mt-16 blur-3xl" />
              <h3 className="text-xs font-bold text-olea-orange uppercase tracking-[0.2em] mb-4 flex items-center gap-2">
                <TrendingUp className="w-4 h-4" />
                Strategic Upsell
              </h3>
              <div className="space-y-4">
                {result?.upsell_suggestions && result.upsell_suggestions.length > 0 ? (
                  result.upsell_suggestions.map((s, i) => (
                    <div key={i} className="bg-white/5 border border-white/10 rounded-xl p-4">
                      <p className="text-xs text-slate-300 leading-relaxed">
                        <Sparkles className="w-4 h-4 text-olea-yellow inline mr-2 mb-1" />
                        {s}
                      </p>
                    </div>
                  ))
                ) : (
                  <div className="bg-white/5 border border-white/10 rounded-xl p-4">
                    <p className="text-xs text-slate-300/50 leading-relaxed italic">
                      Run analysis to see upsell recommendations
                    </p>
                  </div>
                )}

                {/* Quick simulate buttons */}
                {result && (
                  <div className="space-y-2 pt-2">
                    <div className="text-[10px] text-olea-orange/60 uppercase font-bold tracking-widest">Quick Simulations</div>
                    <button
                      onClick={() => handleSimulate('Estimated_Annual_Income', customer.Estimated_Annual_Income * 1.5)}
                      className="w-full py-2 bg-white/5 hover:bg-white/10 text-white/80 text-[10px] font-bold uppercase tracking-widest rounded-lg transition-colors border border-white/10 flex items-center justify-center gap-2"
                    >
                      <FlaskConical className="w-3 h-3" /> +50% Income
                    </button>
                    <button
                      onClick={() => handleSimulate('Deductible_Tier', 'Tier_1_High_Ded')}
                      className="w-full py-2 bg-white/5 hover:bg-white/10 text-white/80 text-[10px] font-bold uppercase tracking-widest rounded-lg transition-colors border border-white/10 flex items-center justify-center gap-2"
                    >
                      <FlaskConical className="w-3 h-3" /> Max Deductible
                    </button>
                    <button
                      onClick={() => handleSimulate('Vehicles_on_Policy', Math.min(customer.Vehicles_on_Policy + 2, 8))}
                      className="w-full py-2 bg-white/5 hover:bg-white/10 text-white/80 text-[10px] font-bold uppercase tracking-widest rounded-lg transition-colors border border-white/10 flex items-center justify-center gap-2"
                    >
                      <FlaskConical className="w-3 h-3" /> +2 Vehicles
                    </button>
                  </div>
                )}
              </div>
            </section>

            {/* Simulation Result Card */}
            <AnimatePresence>
              {simResult && (
                <motion.section
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -10 }}
                  className="glass rounded-2xl p-6 neo-border bg-amber-50/80 border-amber-200"
                >
                  <h3 className="text-xs font-black text-amber-700 uppercase tracking-[0.2em] mb-4 flex items-center gap-2">
                    <FlaskConical className="w-4 h-4" />
                    Simulation Result
                  </h3>
                  <div className="space-y-3">
                    <div className="flex justify-between items-center">
                      <span className="text-[10px] text-amber-600 uppercase font-bold">Base</span>
                      <span className="text-xs font-black text-olea-dark">{bundleName(simResult.base.predicted_bundle)}</span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-[10px] text-amber-600 uppercase font-bold">Modified</span>
                      <span className="text-xs font-black text-olea-dark">{bundleName(simResult.modified.predicted_bundle)}</span>
                    </div>
                    {simResult.delta.bundle_changed && (
                      <div className="bg-amber-100 rounded-lg p-3 text-[10px] font-bold text-amber-800 uppercase tracking-wider text-center">
                        ⚡ Bundle Changed!
                      </div>
                    )}
                    <div className="flex justify-between items-center pt-2 border-t border-amber-200">
                      <span className="text-[10px] text-amber-600 uppercase font-bold">Probability Shift</span>
                      <span className={`text-xs font-mono font-bold ${simResult.delta.top_probability_shift.delta > 0 ? 'text-emerald-600' : 'text-rose-600'}`}>
                        {simResult.delta.top_probability_shift.delta > 0 ? '+' : ''}{pct(simResult.delta.top_probability_shift.delta)}
                      </span>
                    </div>
                  </div>
                </motion.section>
              )}
            </AnimatePresence>
          </div>

          {/* ═══════════════════════════════════════════════════════════
              MIDDLE COLUMN: Decision Timeline & Results
          ═══════════════════════════════════════════════════════════ */}
          <div className="lg:col-span-6 space-y-8">
            {/* Main Prediction Display */}
            <section className="glass rounded-[2.5rem] p-10 neo-border relative overflow-hidden min-h-[500px] flex flex-col justify-center bg-white shadow-2xl shadow-olea-dark/5">
              <AnimatePresence mode="wait">
                {!result && !loading ? (
                  <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    className="text-center space-y-6"
                  >
                    <div className="w-24 h-24 bg-olea-bg rounded-full flex items-center justify-center mx-auto mb-8 border border-olea-dark/5">
                      <Brain className="w-12 h-12 text-olea-terracotta/30" />
                    </div>
                    <h2 className="text-3xl font-serif font-bold text-olea-dark">Awaiting Cognitive Input</h2>
                    <p className="text-slate-400 max-w-sm mx-auto text-sm leading-relaxed">
                      Configure customer profile and click <strong>RUN COGNITIVE ANALYSIS</strong> to engage the multi-agent prediction system.
                    </p>
                  </motion.div>
                ) : loading || step < 5 ? (
                  <motion.div
                    key="loading"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    className="space-y-16"
                  >
                    <div className="text-center">
                      <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-olea-orange/10 border border-olea-orange/20 text-olea-orange text-[10px] font-black uppercase tracking-[0.2em] mb-6">
                        <RefreshCw className="w-3 h-3 animate-spin" />
                        Agent Committee Deliberating
                      </div>
                      <h2 className="text-4xl font-serif font-bold text-olea-dark">Synthesizing Decision Space</h2>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6 max-w-2xl mx-auto">
                      {result?.agent_votes ? (
                        <>
                          {result.agent_votes.map((av, i) => (
                            <AgentNode
                              key={av.agent}
                              name={av.agent}
                              vote={step >= i + 1 ? av.top_bundle_name : 'Analyzing...'}
                              active={step >= i + 1}
                              delay={i * 0.1}
                            />
                          ))}
                          <AgentNode
                            name="Meta Decision Engine"
                            vote={step >= 5 ? `${result.agent_consensus.top_bundle_name} (${result.agent_consensus.agreement}/${result.agent_consensus.total_agents} agree)` : 'Fusing Probabilities...'}
                            active={step >= 5}
                            delay={0.4}
                          />
                        </>
                      ) : (
                        <>
                          <AgentNode name="Life-Stage Agent" vote="Analyzing..." active={step >= 1} delay={0.1} />
                          <AgentNode name="Financial Agent" vote="Analyzing..." active={step >= 2} delay={0.2} />
                          <AgentNode name="Stability Agent" vote="Analyzing..." active={step >= 3} delay={0.3} />
                          <AgentNode name="Friction Agent" vote="Analyzing..." active={step >= 4} delay={0.4} />
                        </>
                      )}
                    </div>
                  </motion.div>
                ) : (
                  <motion.div
                    key="result"
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="space-y-10"
                  >
                    <div className="text-center">
                      <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-olea-dark text-white text-[10px] font-black uppercase tracking-[0.2em] mb-6">
                        <Target className="w-3 h-3 text-olea-orange" />
                        Optimal Bundle Alignment
                      </div>
                      <h2 className="text-5xl lg:text-6xl font-serif font-black text-olea-dark mb-4 leading-tight">
                        {bundleName(result!.predicted_bundle)}
                      </h2>
                      <div className="flex items-center justify-center gap-4">
                        <span className="px-4 py-1 bg-olea-orange/10 text-olea-orange text-xs font-black uppercase rounded-full">
                          Bundle #{result!.predicted_bundle}
                        </span>
                      </div>
                    </div>

                    <div className="grid grid-cols-3 gap-8 border-y border-olea-dark/5 py-10">
                      <div className="text-center space-y-3">
                        <div className="text-[10px] text-slate-400 uppercase font-black tracking-widest">Confidence</div>
                        <div className="text-4xl font-mono font-bold text-olea-dark">{pct(result!.confidence)}</div>
                        <div className="w-full h-1.5 bg-slate-100 rounded-full overflow-hidden">
                          <motion.div
                            initial={{ width: 0 }}
                            animate={{ width: pct(result!.confidence) }}
                            className="h-full olea-gradient"
                          />
                        </div>
                      </div>
                      <div className="text-center space-y-3 border-x border-olea-dark/5 px-4">
                        <div className="text-[10px] text-slate-400 uppercase font-black tracking-widest">Uncertainty</div>
                        <div className={`text-3xl font-mono font-bold ${
                          result!.uncertainty.level === 'Low' ? 'text-emerald-600' :
                          result!.uncertainty.level === 'Moderate' ? 'text-amber-600' : 'text-rose-600'
                        }`}>
                          {result!.uncertainty.level}
                        </div>
                        <div className="text-[10px] text-slate-400 italic">H = {result!.uncertainty.entropy.toFixed(3)}</div>
                      </div>
                      <div className="text-center space-y-3">
                        <div className="text-[10px] text-slate-400 uppercase font-black tracking-widest">Agent Consensus</div>
                        <div className="text-4xl font-mono font-bold text-olea-dark">
                          {result!.agent_consensus.agreement}/{result!.agent_consensus.total_agents}
                        </div>
                        <div className="text-[10px] text-slate-400 italic">Agents Agree</div>
                      </div>
                    </div>

                    {/* Narrative Timeline */}
                    {result!.narrative.decision_timeline.length > 0 && (
                      <div className="space-y-4">
                        <h4 className="text-[10px] text-slate-400 uppercase font-black tracking-widest flex items-center gap-2">
                          <MessageSquareQuote className="w-3 h-3" /> Decision Narrative
                        </h4>
                        <div className="space-y-3">
                          {result!.narrative.decision_timeline.map((ns) => (
                            <motion.div
                              key={ns.step}
                              initial={{ opacity: 0, x: -10 }}
                              animate={{ opacity: 1, x: 0 }}
                              transition={{ delay: ns.step * 0.1 }}
                              className="flex gap-4 items-start"
                            >
                              <div className="w-8 h-8 rounded-full bg-olea-orange/10 flex items-center justify-center flex-shrink-0 mt-0.5">
                                <span className="text-[10px] font-black text-olea-orange">{ns.step}</span>
                              </div>
                              <div className="flex-1 min-w-0">
                                <div className="flex items-center gap-2">
                                  <span className="text-[10px] font-black text-olea-terracotta uppercase tracking-wider">{ns.agent}</span>
                                  <span className="text-xs font-semibold text-olea-dark">{ns.title}</span>
                                </div>
                                <p className="text-xs text-slate-500 leading-relaxed mt-1">{ns.summary}</p>
                              </div>
                            </motion.div>
                          ))}
                        </div>
                      </div>
                    )}
                  </motion.div>
                )}
              </AnimatePresence>
            </section>

            {/* Probability Distribution */}
            <section className="glass rounded-3xl p-8 neo-border bg-white/90">
              <h3 className="text-xs font-black text-olea-dark uppercase tracking-[0.2em] mb-8 flex items-center gap-3">
                <BarChart3 className="w-4 h-4 text-olea-orange" />
                Bundle Probability Distribution
              </h3>
              {result ? (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-x-12 gap-y-6">
                  {result.probabilities
                    .map((p, i) => ({ name: bundleName(i), prob: p, idx: i }))
                    .sort((a, b) => b.prob - a.prob)
                    .map((p, rank) => (
                      <div key={p.idx} className="space-y-2">
                        <div className="flex justify-between text-[10px] font-black uppercase tracking-widest">
                          <span className="text-slate-600 truncate mr-2">{p.name}</span>
                          <span className="text-olea-terracotta flex-shrink-0">{pct(p.prob)}</span>
                        </div>
                        <div className="w-full h-1.5 bg-slate-100 rounded-full overflow-hidden">
                          <motion.div
                            initial={{ width: 0 }}
                            animate={{ width: `${Math.max(p.prob * 100, 0.5)}%` }}
                            transition={{ delay: rank * 0.05 }}
                            className={`h-full ${rank === 0 ? 'olea-gradient' : 'bg-slate-300'}`}
                          />
                        </div>
                      </div>
                    ))}
                </div>
              ) : (
                <div className="text-center py-12 text-slate-400 italic text-xs">Run analysis to see probability distribution</div>
              )}
            </section>
          </div>

          {/* ═══════════════════════════════════════════════════════════
              RIGHT COLUMN: Sensitivity, Similarity & Insights
          ═══════════════════════════════════════════════════════════ */}
          <div className="lg:col-span-3 space-y-8">
            {/* Sensitivity Engine */}
            <section className="glass rounded-2xl p-6 neo-border bg-white/90">
              <h3 className="text-xs font-black text-olea-dark uppercase tracking-[0.2em] mb-8 flex items-center gap-3 border-b border-olea-dark/5 pb-4">
                <Activity className="w-4 h-4 text-olea-orange" />
                Sensitivity Engine
              </h3>
              <div className="space-y-6">
                {result?.top_influences && result.top_influences.length > 0 ? (
                  result.top_influences.slice(0, 6).map((inf) => (
                    <div key={inf.feature} className="space-y-2">
                      <div className="flex items-center justify-between">
                        <span className="text-[10px] font-black text-olea-dark uppercase tracking-widest truncate mr-2">{inf.label}</span>
                        {inf.direction === 'positive' ? (
                          <ArrowUpRight className="w-4 h-4 text-emerald-500 flex-shrink-0" />
                        ) : (
                          <ArrowDownRight className="w-4 h-4 text-rose-500 flex-shrink-0" />
                        )}
                      </div>
                      <div className="flex items-center gap-3">
                        <div className="flex-1 h-1 bg-slate-100 rounded-full overflow-hidden">
                          <motion.div
                            initial={{ width: 0 }}
                            animate={{ width: `${Math.min(Math.abs(inf.impact) * 200, 100)}%` }}
                            className={`h-full ${inf.direction === 'positive' ? 'bg-emerald-500' : 'bg-rose-500'}`}
                          />
                        </div>
                        <span className="text-[10px] font-mono font-bold text-slate-400">{Math.abs(inf.impact).toFixed(3)}</span>
                      </div>
                    </div>
                  ))
                ) : (
                  <div className="text-center py-12 text-slate-400 italic text-xs">Inference trace required...</div>
                )}
              </div>
            </section>

            {/* Bundle Similarity */}
            <section className="glass rounded-2xl p-6 neo-border bg-white/90">
              <h3 className="text-xs font-black text-olea-dark uppercase tracking-[0.2em] mb-6 flex items-center gap-3">
                <Layers className="w-4 h-4 text-olea-orange" />
                Bundle Similarity
              </h3>
              <div className="space-y-3">
                {result?.nearest_bundles && result.nearest_bundles.length > 0 ? (
                  result.nearest_bundles.map((b) => (
                    <div
                      key={b.bundle}
                      className="p-4 rounded-xl bg-olea-bg border border-olea-dark/5 group hover:border-olea-orange/30 transition-all cursor-default"
                    >
                      <div className="flex items-center justify-between">
                        <div className="min-w-0 mr-2">
                          <div className="text-[10px] font-black text-olea-dark uppercase tracking-widest truncate">{b.bundle_name}</div>
                          <div className="text-[10px] text-slate-400 mt-1 leading-relaxed">{b.similarity_context}</div>
                        </div>
                        <div
                          className={`text-[8px] font-black px-2 py-1 rounded-full uppercase tracking-tighter flex-shrink-0 ${
                            b.similarity_level === 'Very similar'
                              ? 'bg-emerald-100 text-emerald-700'
                              : b.similarity_level === 'Moderately similar'
                              ? 'bg-amber-100 text-amber-700'
                              : 'bg-slate-100 text-slate-600'
                          }`}
                        >
                          {b.similarity_level}
                        </div>
                      </div>
                    </div>
                  ))
                ) : (
                  <div className="text-center py-8 text-slate-400 italic text-xs">Run analysis to see similar bundles</div>
                )}
              </div>
            </section>

            {/* Business Insight */}
            <section className="bg-olea-terracotta rounded-2xl p-6 shadow-2xl text-white relative overflow-hidden">
              <div className="absolute bottom-0 left-0 w-full h-1 bg-olea-orange" />
              <h3 className="text-[10px] font-black uppercase tracking-[0.3em] mb-6 flex items-center gap-2 text-white/60">
                <Info className="w-4 h-4" />
                Executive Insight
              </h3>
              {result?.business_insight ? (
                <div className="space-y-5">
                  <div className="flex justify-between items-center border-b border-white/10 pb-3">
                    <span className="text-[10px] uppercase font-bold text-white/60">Segment</span>
                    <span className="text-xs font-black">{result.business_insight.customer_segment}</span>
                  </div>
                  <div className="flex justify-between items-center border-b border-white/10 pb-3">
                    <span className="text-[10px] uppercase font-bold text-white/60">Retention Risk</span>
                    <span className={`text-xs font-black ${
                      result.business_insight.retention_risk.includes('Low') ? 'text-emerald-400' :
                      result.business_insight.retention_risk.includes('Moderate') ? 'text-amber-400' : 'text-rose-400'
                    }`}>
                      {result.business_insight.retention_risk}
                    </span>
                  </div>
                  <div className="flex justify-between items-start">
                    <span className="text-[10px] uppercase font-bold text-white/60 flex-shrink-0 mr-2">Upsell</span>
                    <span className="text-xs font-black text-right">{result.business_insight.upsell_potential}</span>
                  </div>
                </div>
              ) : (
                <div className="space-y-5">
                  <div className="flex justify-between items-center border-b border-white/10 pb-3">
                    <span className="text-[10px] uppercase font-bold text-white/60">Segment</span>
                    <span className="text-xs font-medium text-white/30 italic">—</span>
                  </div>
                  <div className="flex justify-between items-center border-b border-white/10 pb-3">
                    <span className="text-[10px] uppercase font-bold text-white/60">Retention Risk</span>
                    <span className="text-xs font-medium text-white/30 italic">—</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-[10px] uppercase font-bold text-white/60">Upsell</span>
                    <span className="text-xs font-medium text-white/30 italic">—</span>
                  </div>
                </div>
              )}
            </section>

            {/* Agent Votes Panel */}
            {result?.agent_votes && (
              <motion.section
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className="glass rounded-2xl p-6 neo-border bg-white/90"
              >
                <h3 className="text-xs font-black text-olea-dark uppercase tracking-[0.2em] mb-4 flex items-center gap-3">
                  <Brain className="w-4 h-4 text-olea-orange" />
                  Agent Votes
                </h3>
                <div className="space-y-4">
                  {result.agent_votes.map((av) => (
                    <div key={av.agent} className="space-y-1">
                      <div className="flex items-center justify-between">
                        <span className="text-[10px] font-black text-olea-dark uppercase tracking-widest">{av.agent}</span>
                        <span className="text-[10px] font-mono text-olea-terracotta">{pct(av.confidence)}</span>
                      </div>
                      <div className="text-xs text-slate-600 font-medium">{av.top_bundle_name}</div>
                      <div className="w-full h-1 bg-slate-100 rounded-full overflow-hidden">
                        <div className="h-full olea-gradient" style={{ width: pct(av.confidence) }} />
                      </div>
                    </div>
                  ))}
                  <div className="pt-3 border-t border-olea-dark/5">
                    <div className="flex justify-between text-[10px] font-black uppercase tracking-widest">
                      <span className="text-olea-dark">Consensus</span>
                      <span className="text-olea-orange">{result.agent_consensus.top_bundle_name}</span>
                    </div>
                  </div>
                </div>
              </motion.section>
            )}
          </div>
        </main>

        {/* Footer */}
        <footer className="mt-16 pt-10 border-t border-olea-dark/10 flex flex-col md:flex-row justify-between items-center gap-8">
          <div className="flex items-center gap-10">
            <div className="flex items-center gap-3">
              <div className="w-2 h-2 rounded-full bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.5)]" />
              <span className="text-[10px] font-black text-olea-dark/40 uppercase tracking-[0.2em]">Neural Engine Active</span>
            </div>
            <div className="flex items-center gap-3">
              <Lock className="w-3 h-3 text-olea-dark/30" />
              <span className="text-[10px] font-black text-olea-dark/40 uppercase tracking-[0.2em]">LightGBM Multi-Agent</span>
            </div>
          </div>

          <div className="text-[10px] font-black text-olea-dark/40 uppercase tracking-[0.2em]">
            © 2025 OLEA INSURANCE SOLUTIONS AFRICA
          </div>
          <div className="flex items-center gap-6">
            <div className="h-4 w-px bg-olea-dark/10" />
            <div className="flex gap-4">
              <div className="w-6 h-6 rounded-full bg-olea-dark/5 flex items-center justify-center text-olea-dark/40 hover:bg-olea-orange hover:text-white transition-all cursor-pointer">
                <Users className="w-3 h-3" />
              </div>
              <div className="w-6 h-6 rounded-full bg-olea-dark/5 flex items-center justify-center text-olea-dark/40 hover:bg-olea-orange hover:text-white transition-all cursor-pointer">
                <Activity className="w-3 h-3" />
              </div>
            </div>
          </div>
        </footer>
      </div>
    </div>
  );
}
