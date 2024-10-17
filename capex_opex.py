"""
https://www.nrel.gov/docs/fy22osti/83586.pdf
MMP:    modeled market price
MSP:    minimum sustainable price
ILR:	inverter loading ratio
        This ratio of PV to inverter power is measured as the DC/AC ratio. A healthy design will typically have a DC/AC ratio of 1.25. The reason for this is that about less than 1% of the energy produced by the PV array throughout its life will be at a power above 80% capacity  
PII:    Permitting, Inspection, Interconnection
"""

def pv_capex_func(pv_size_kw, mode='mmp'):
    if mode == 'msp':
        module_efficiency = 0.203
        watt_per_m2 = 1000
        module_price = 0.48
        inverter_price = 0.36
        ilr = 1.21
        structual_bos = 19.1
        electrical_bos_per_watt = 37.2
        electrical_bos_init = 1016
        sales_tax = 0.051
        construction_labor_hour_per_m2 = 0.56
        construction_labor_per_hour = 24
        electrician_labor_hour_per_m2 = 0.51
        electrician_labor_per_hour = 38.15
        pii = 1628
        sales_marketing = 3139
        overhead = 2060
        profit = 0.17
    elif mode == 'mmp':
        module_efficiency = 0.203
        watt_per_m2 = 1000
        module_price = 0.54
        inverter_price = 0.53
        ilr = 1.21
        structual_bos = 31.5
        electrical_bos_per_watt = 43.7
        electrical_bos_init = 1231
        sales_tax = 0.051
        construction_labor_hour_per_m2 = 0.56
        construction_labor_per_hour = 24
        electrician_labor_hour_per_m2 = 0.51
        electrician_labor_per_hour = 38.15
        pii = 1628
        sales_marketing = 3139
        overhead = 2060
        profit = 0.17
    else:
        raise ValueError('Invalid mode')
    
    pv_size_m2 = pv_size_kw * 1000 / module_efficiency / watt_per_m2
    module_cost = module_price * pv_size_kw * 1000
    inverter_cost = inverter_price * pv_size_kw / ilr * 1000
    structual_bos_cost = structual_bos * pv_size_m2
    electrical_bos_cost = electrical_bos_per_watt * pv_size_m2 + electrical_bos_init
    sales_tax_cost = sales_tax * (module_cost + inverter_cost + structual_bos_cost + electrical_bos_cost)
    construction_labor_cost = construction_labor_hour_per_m2 * pv_size_m2 * construction_labor_per_hour
    electrician_labor_cost = electrician_labor_hour_per_m2 * pv_size_m2 * electrician_labor_per_hour
    profit_cost = profit * (module_cost + inverter_cost + structual_bos_cost + electrical_bos_cost + sales_tax_cost + construction_labor_cost + electrician_labor_cost + pii)
    capex = module_cost + inverter_cost + structual_bos_cost + electrical_bos_cost + sales_tax_cost + construction_labor_cost + electrician_labor_cost + pii + sales_marketing + overhead + profit_cost
    return capex


def pv_opex_func(pv_size_kw, mode='mmp'):
    if mode == 'msp':
        opex_per_kwdc = 29.49
        opex = opex_per_kwdc * pv_size_kw
    elif mode == 'mmp':
        opex_per_kwdc = 31.12
        opex = opex_per_kwdc * pv_size_kw
    else:
        raise ValueError('Invalid mode')
    return opex


def battery_capex(battery_size_kwh, pv_size_kw, mode='mmp'):
    battery_pack_footprint_kWh_per_m2 = 8.3
    if mode == 'msp':
        battery_price_per_kwh = 235
        battery_price = battery_price_per_kwh * battery_size_kwh
        battery_based_inverter_cost = 0.23 * pv_size_kw * 1000
        BOS_cost = 1362
        supply_chain_cost = 0.065 * (battery_price + battery_based_inverter_cost + BOS_cost)
        engineering_fee = 95
        pii = 1633  # pii includes permitting cost $286
        permitting_cost = 286
        sales_tax = 0.051 * (battery_price + battery_based_inverter_cost + BOS_cost + permitting_cost)
        direct_labor_cost = 20.8 * battery_size_kwh / battery_pack_footprint_kWh_per_m2 * 34.7
        sales_and_marketing = 3851
        overhead = 2285
        profit = 0.17 * (battery_price + battery_based_inverter_cost + BOS_cost + supply_chain_cost + sales_tax + direct_labor_cost)
        capex = battery_price + battery_based_inverter_cost + BOS_cost + supply_chain_cost + engineering_fee + pii + sales_tax + direct_labor_cost + sales_and_marketing + overhead + profit
    elif mode == 'mmp':
        battery_price_per_kwh = 283
        battery_price = battery_price_per_kwh * battery_size_kwh
        battery_based_inverter_cost = 0.29 * pv_size_kw * 1000
        BOS_cost = 1567
        supply_chain_cost = 0.065 * (battery_price + battery_based_inverter_cost + BOS_cost)
        engineering_fee = 95
        pii = 1633  # pii includes permitting cost $286
        permitting_cost = 286
        sales_tax = 0.051 * (battery_price + battery_based_inverter_cost + BOS_cost + permitting_cost)
        direct_labor_cost = 20.8 * battery_size_kwh / battery_pack_footprint_kWh_per_m2 * 34.7
        sales_and_marketing = 3851
        overhead = 2285
        profit = 0.17 * (battery_price + battery_based_inverter_cost + BOS_cost + supply_chain_cost + sales_tax + direct_labor_cost)
        capex = battery_price + battery_based_inverter_cost + BOS_cost + supply_chain_cost + engineering_fee + pii + sales_tax + direct_labor_cost + sales_and_marketing + overhead + profit
    else:
        raise ValueError('Invalid mode')
    # print(battery_price, battery_based_inverter_cost, BOS_cost, supply_chain_cost, sales_tax, direct_labor_cost, engineering_fee, pii, sales_and_marketing, overhead+profit)
    return capex


def battery_opex(mode='mmp'):
    """
    battery opex is not provided in the NREL report
    """
    return 0


if __name__ == "__main__":
    mode = 'mmp'

    pv_size_kw = 8.0
    pv_capex = pv_capex_func(pv_size_kw, mode)
    print(f'PV Capex: {pv_capex}')
    pv_opex = pv_opex_func(pv_size_kw, mode)
    print(f'PV Opex: {pv_opex}')

    battery_size = 13.5
    battery_capex = battery_capex(battery_size, pv_size_kw, mode)
    print(f'Battery Capex: {battery_capex}')

